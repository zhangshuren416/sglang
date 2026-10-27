# SPDX-License-Identifier: Apache-2.0
"""Protect raw-layout boundaries and cross-forward SDMA slot ownership."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.moe.dfc_config import (
    WeightPoolConfig,
    physical_layout,
    pool_capacity,
)
from sglang.srt.hardware_backend.npu.moe.dfc_offload import DfcOffloader
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Stream:
    def __init__(self, device):
        self.device = device
        self.commands = []

    def enqueue(self, operation):
        self.commands.append(operation)

    def wait_event(self, event):
        self.enqueue(lambda: event.done)

    def wait_stream(self, stream):
        event = self.device.Event()
        event.record(stream)
        self.wait_event(event)

    def synchronize(self):
        self.device.drain()


class _Event:
    def __init__(self, device):
        self.device = device
        self.done = False

    def record(self, stream):
        def complete():
            self.done = True
            return True

        stream.enqueue(complete)

    def query(self):
        return self.done


class _Device:
    """Two asynchronous queues; copy always gets the first chance to progress.

    Omitting a consumed dependency deterministically overwrites a queued reader.
    No GPU driver, native extension, or model execution is involved.
    """

    def __init__(self):
        self.compute = _Stream(self)
        self.computes = [self.compute]
        self.copy = _Stream(self)
        self.active = self.compute

    def current_stream(self, _device):
        return self.compute

    def Event(self):
        return _Event(self)

    def stream(self, stream):
        device = self

        class Scope:
            def __enter__(self):
                self.previous = device.active
                device.active = stream

            def __exit__(self, *_):
                device.active = self.previous

        return Scope()

    def drain(self):
        streams = (self.copy, *self.computes)
        while any(stream.commands for stream in streams):
            progress = False
            for stream in streams:
                if stream.commands and stream.commands[0]():
                    stream.commands.pop(0)
                    progress = True
            if not progress:
                raise AssertionError("SDMA/compute dependency cycle")


class _Plan:
    def __init__(self, device, slot, value):
        self.device, self.slot, self.value = device, slot, value

    def enqueue(self, submitted):
        submitted.record(self.device.copy)

        def copy():
            self.slot[0] = self.value
            return True

        self.device.copy.enqueue(copy)


class _Moe(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

    def forward(self, enqueue):
        enqueue()
        return object()


class _Decoder(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.mlp = _Moe(layer_id)
        self.fail_attention = False

    def forward(self, enqueue):
        if self.fail_attention:
            raise RuntimeError("attention failed")
        return self.mlp(enqueue)


class TestDfcWeightOffload(CustomTestCase):
    def test_pool_capacity_and_native_configuration(self):
        """Native reserve uses whole GiB, not logical payload bytes or JSON booleans."""
        gib = 1 << 30
        self.assertEqual(pool_capacity(gib + 1, "auto"), 2 * gib)
        with self.assertRaises(ValueError):
            pool_capacity(gib + 1, gib)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.json"
            config = {
                "schema_version": 1,
                "pool_id": "test",
                "store_url": "tcp://host:1234",
            }
            invalid_fields = (
                ("sdma_channels", True),
                ("reserve_bytes_per_rank", False),
            )
            for field, value in invalid_fields:
                path.write_text(json.dumps({**config, field: value}))
                with self.assertRaises(ValueError):
                    WeightPoolConfig.read(str(path))
            path.write_text(json.dumps(config))
            self.assertEqual(WeightPoolConfig.read(str(path)).sdma_channels, 4)

    def test_storage_views_cannot_be_exported_as_owning_nz(self):
        """A view's logical shape must not authorize a DMA of the entire allocation."""
        owner = torch.empty(32, dtype=torch.int8)
        self.assertEqual(physical_layout(owner, 2)[-1], 32)
        with self.assertRaises(ValueError):
            physical_layout(owner[1:], 2)
        with self.assertRaises(ValueError):
            physical_layout(owner[:1], 2)

    def test_odd_even_layers_across_queued_forwards(self):
        """Tail readers must finish before the next forward overwrites a slot."""
        for count in (1, 2, 3, 4, 5):
            with self.subTest(layers=count):
                device = _Device()
                slots = [[None], [None]]
                observed = []
                with get_context().override_server_args():
                    offloader = DfcOffloader()
                offloader.device = "fake"
                offloader.stream = device.copy
                offloader.ready_for_forward = True
                # Absolute decoder IDs differ from sparse-layer/slot indices.
                offloader.layers = [_Decoder(3 + 2 * i) for i in range(count)]
                for layer in offloader.layers:
                    offloader._install_hooks(layer)
                offloader.plans = {
                    i: _Plan(device, slots[(i - 1) % 2], i) for i in range(1, count)
                }
                with patch.object(torch, "npu", device, create=True):
                    for epoch in range(8):
                        if epoch == 4:
                            # Warmup and serving use different streams, still serially.
                            device.compute = _Stream(device)
                            device.computes.append(device.compute)
                        for index in range(count):
                            def read(index=index, epoch=epoch):
                                value = 0 if index == 0 else slots[(index - 1) % 2][0]
                                self.assertEqual(value, index)
                                observed.append((epoch, value))
                                return True

                            offloader.layers[index](
                                lambda read=read: device.compute.enqueue(read)
                            )
                    device.drain()
                for hook in offloader.hooks:
                    hook.remove()
                self.assertEqual(
                    observed, [(e, i) for e in range(8) for i in range(count)]
                )

    def test_failed_consumer_cannot_restart_the_ring(self):
        with get_context().override_server_args():
            offloader = DfcOffloader()
        device = _Device()
        offloader.device = "fake"
        offloader.stream = device.copy
        offloader.ready_for_forward = True
        offloader.layers = [
            SimpleNamespace(layer_id=0, mlp=SimpleNamespace(alt_stream=None))
        ]
        with patch.object(torch, "npu", device, create=True):
            offloader.on_layer_enter(0)
            with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                with offloader.moe_scope(0):
                    raise RuntimeError("consumer failed")
            with self.assertRaises(RuntimeError):
                offloader.on_layer_enter(0)

    def test_hook_failures_stop_reuse_before_or_inside_moe(self):
        """always_call hooks must poison the ring when an attention/MLP call raises."""
        for where in ("attention", "moe"):
            with self.subTest(where=where):
                with get_context().override_server_args():
                    offloader = DfcOffloader()
                device = _Device()
                offloader.device = "fake"
                offloader.stream = device.copy
                offloader.ready_for_forward = True
                layer = _Decoder(3)
                layer.fail_attention = where == "attention"
                offloader.layers = [layer]
                offloader._install_hooks(layer)

                def fail():
                    raise RuntimeError("moe failed")

                try:
                    with patch.object(torch, "npu", device, create=True):
                        with self.assertRaisesRegex(RuntimeError, f"{where} failed"):
                            layer(fail)
                        self.assertTrue(offloader.failed)
                        self.assertIsNone(offloader.moe_context)
                        with self.assertRaises(RuntimeError):
                            layer(lambda: None)
                finally:
                    for hook in offloader.hooks:
                        hook.remove()


if __name__ == "__main__":
    unittest.main()
