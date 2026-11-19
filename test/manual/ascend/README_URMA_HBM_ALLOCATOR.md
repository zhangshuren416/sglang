# Ascend URMA HBM allocator 验证

在能正常使用 `torch_npu` 的 A5 环境下运行 [手测脚本](test_urma_hbm_allocator.py)。MemFabric wheel 需要包含 `ShareMemory.symmetric_size`，双卡用例还需要 `UrmaCopyPlan`。脚本默认使用第一张或前两张可见卡，可通过 `ASCEND_RT_VISIBLE_DEVICES` 选卡。

## 运行

在 SGLang 仓库根目录执行一次，确保使用当前工作区里的 allocator。

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

**单卡本地验证。** 使用 `shm` 的 MTE 模式建 HBM 窗口，无需初始化 DRAM offload。对 ND 和 `FRACTAL_NZ` tensor 执行 `copy_()`，检查地址保持不变，并用 NPU 矩阵乘核对输出。

```bash
python3 test/manual/ascend/test_urma_hbm_allocator.py -v \
  TestUrmaHbmAllocator.test_local_copy_and_compute
```

**双卡 URMA 验证。** 使用两张互通的 A5，测试双向整块搬运和 `byte_view()` 分片搬运，再用 NPU 矩阵乘核对输出。每个 rank 创建 1 GiB HBM 窗口，并按当前 `UrmaCopyPlan` 的初始化要求创建 1 GiB 本地 DRAM 池。权重从对端 HBM 直接搬到本地 HBM。

```bash
python3 test/manual/ascend/test_urma_hbm_allocator.py -v \
  TestUrmaHbmAllocator.test_dual_rank_copy_and_compute
```

去掉末尾的测试名即可运行两个用例。

## 判断结果

单卡应分别打印 `[local] ACL_FORMAT_ND ... passed` 和 `[local] ACL_FORMAT_FRACTAL_NZ ... passed`。双卡的两个 rank 都应打印两种格式的 `whole and split copies passed`。最终 `unittest` 应显示 `OK`。

## 权重槽布局验证

[槽布局脚本](test_urma_weight_slot.py) 在正式实现共享管理器前验证三个设备行为：同一个 `UrmaCopyPlan` 跨轮重复 `enqueue()`、本地 D2D 填充和远端 URMA 搬运写入同一个槽的不同 expert 区间、以及 MXFP8 的 `FRACTAL_NZ` + transpose 视图按全量 expert 分配并按 expert 字节偏移填充。

需要两张互通的 A5。脚本每次 spawn 两个进程，每卡各建 1 GiB HBM 窗口和 1 GiB 本地 DRAM 池；不要在已跑 sglang server 的进程里执行，脚本会 `shm.create(id=0)`，同进程有别的 shm 用户会冲突。

先检查环境：

```bash
cd /path/to/sglang
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1

python3 - <<'PY'
import torch, torch_npu
from memfabric_hybrid import offload, shm
print("npu count:", torch.npu.device_count())
print("symmetric_size:", hasattr(shm.ShareMemory, "symmetric_size"))
print("UrmaCopyPlan:", hasattr(offload, "UrmaCopyPlan"))
PY
```

`npu count` 要 ≥ 2，后两项都要 `True`。

运行全部用例：

```bash
python3 test/manual/ascend/test_urma_weight_slot.py -v
```

单独运行：

```bash
python3 test/manual/ascend/test_urma_weight_slot.py -v TestUrmaWeightSlot.test_plan_reuse
python3 test/manual/ascend/test_urma_weight_slot.py -v TestUrmaWeightSlot.test_slot_merge_nd
python3 test/manual/ascend/test_urma_weight_slot.py -v TestUrmaWeightSlot.test_slot_merge_nz
python3 test/manual/ascend/test_urma_weight_slot.py -v TestUrmaWeightSlot.test_slot_merge_transposed_nz
```

每个 rank 应打印：

```text
[rank 0] UrmaCopyPlan reuse passed
[rank 0] ACL_FORMAT_ND transposed=False slot merge passed (async_local=True)
[rank 0] ACL_FORMAT_FRACTAL_NZ transposed=False slot merge passed (async_local=True)
[rank 0] ACL_FORMAT_FRACTAL_NZ transposed=True slot merge passed (async_local=True)
```

rank 1 同理，最终 `unittest` 显示 `OK`。

- `async_local=True`：本地填充走了 `acl.rt.memcpy_async`。
- `async_local=False`：当前镜像没有该接口或调用失败，回退同步拷贝；正确性仍应通过，但本地填充和计算的重叠需要再确认。
- `transposed view lost the FRACTAL_NZ format`：MXFP8 的 transpose 视图丢了 NZ 格式，方案需要调整。
- `slot bytes differ from the full reference`：槽内合并字节不对，需要检查本地填充、远端搬运或 expert 物理步长。

失败时保留完整 stdout/stderr 和以下信息：

```bash
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
python3 -c "import torch, torch_npu, memfabric_hybrid; print(torch.__version__, torch_npu.__version__, memfabric_hybrid.__file__)"
```

卡住不退出时看是否停在 `sync.wait()`，可以加 `timeout 300`；需要更多 MemFabric 日志时设置 `export MF_LOG_LEVEL=1`。
