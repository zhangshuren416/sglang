# Ascend MemFabric MoE 权重预取

权重搬运与计算后端分别选择：

- `--moe-a2a-backend deepep` 保留原生 DeepEP dispatch、激活、FFN 和 combine。
- `--moe-weight-offload-backend memfabric` 将后续 MoE 专家执行包保存到
  MemFabric SHARED DRAM，并用 SDMA 预取到两个固定 HBM slot。
- `--moe-weight-offload-backend none` 保留全 HBM 对照。

DeepSeek-V4 (`deepseek_v4` / `DeepseekV4DecoderLayer`) 已接入独立权重预取，
无需启用 DFC，也不要求 ZBAL 或 PD 分离。V4 原有 hash router、裁剪、SiTU 等
计算语义和量化布局由原后端处理。当前 **V4 仍不支持 DFC**。

现阶段完成的是源码接入和静态检查；CPU 模拟、A3 精度与性能尚未验证。

## V4-Flash 单机启动

使用现有的 Ascend A3 镜像、ModelSlim W8A8 权重和包含
`memfabric_hybrid.offload.SdmaCopyPlan` 的已安装 wheel。脚本不安装依赖或构建环境。

复制 [moe_weight_pool.example.json](moe_weight_pool.example.json) 到自己的配置路径，
修改 `store_url`，使用与其他作业隔离的地址/端口。
配置只包含 `store_url`、`reserve_bytes` 和 `sdma_channels`；后两项可省略，
默认分别为 `"auto"` 和 `4`。

在仓库根目录运行：

```bash
MODEL_PATH=/path/to/DeepSeek-V4-Flash-w8a8-mtp \
WEIGHT_OFFLOAD_BACKEND=memfabric \
WEIGHT_POOL_CONFIG=/path/to/moe_weight_pool.json \
bash test/manual/ascend/launch_dsv4_flash_deepep_single_node.sh
```

可以追加环境变量 `DRY_RUN=1`，只打印命令，不加载 CANN 或模型。
脚本默认 TP/EP/DP16、DeepEP normal、8192 prefill、16384 context、并发 1。
预取开启时使用独立 shared experts，并关闭图、调度 overlap 和 MTP。
启动和运行日志实时输出并保存在 `./logs/dsv4-flash-deepep/`。

手动接入现有 DeepEP 启动命令时，追加：

```bash
--moe-weight-offload-backend memfabric \
--moe-weight-offload-config /path/to/moe_weight_pool.json
```

参数解析会独立关闭 shared expert fusion、图执行和调度 overlap。
支持 colocated、PD prefill 和普通 decode，均按每次模型前向轮转 slot；
decode 每步仍需搬运权重，需单独测量带宽和延迟。

## 实现与边界

- `weight_offload.py` 直接继承 `BaseOffloader`，不导入 DFC。
  只有显式选择 `ascend_dfc` 时，工厂才附加独立的 `DfcModelAdapter`。
- `weight_offload_config.py` 管理共享池、物理布局和容量。
- 首个实际 MoE 常驻 HBM；后续 routed 专家权重、scale、offset 和直接 Tensor 缓存
  在原生量化后导出。shared experts 始终常驻 HBM，不参与搬运。
  DeepSeek/GLM 前导 dense 层不占用 MoE slot 序号。
- 在 MoE decoder 入口预取下一 MoE；MoE 入口等待 ready；计算、shared 和 routed
  辅助流结束后记录 consumed。复制流等待旧 consumed，避免跨层或跨请求覆盖读者。
- 只搬运权重，Attention、压缩器、IndexCache、mHC 和模型 forward 主体保持原有路径。
- 退出时先排空读者和 SDMA，移除 hooks、关闭计划并释放池，最后销毁模型并行组。

当前支持 Ascend、ModelSlim W8A8、BF16、PP1、MoE TP1、MoE DP1、固定 EP 和 eager。
模型适配包括 V4、Qwen3-MoE、DeepSeek V3/V3.2/R1 和普通 GLM MoE DSA。
V4.1、其他量化方式以及异形专家执行包未接入。
专家执行包必须同 shape/dtype/stride/format/物理大小；路由与激活元数据可以不同。
原始复制支持 ND / FRACTAL_NZ，拒绝偏移 view 和部分 ND allocation。

投机解码、TBO/SBO、CP/DCP、DWDP、EPLB、冗余/弹性专家、LoRA、memory saver、
通用 CPU offloader、CPU 权重备份、torch.compile、运行期 PD 角色切换和 remote/IPC
加载暂未接入。MemFabric HiCache 不能与权重池占用同一个 native entity。
在线权重更新、检查点导出与按名读取共享 slot 中的原始权重会被拒绝。

共享池由 TP rank 0 唯一分配物理 DRAM，其他 rank 的 `alloc_size=0`。
所有 rank 的 `reserve_size` 相同，用于建立一致的虚拟地址跨度；这不等于重复分配物理内存。
rank 0 分配一个连续 arena 并广播基址，各 rank 写入独立的 EP 分片区域，预取计划直接使用共享 GVA。
只有 rank 0 释放 arena，所有 rank 关闭计划后才销毁共享池。

`reserve_bytes="auto"` 按所有 rank 的卸载权重总和计算，并向上对齐 GiB。
手填正整数表示整个共享池的容量，单位为字节，不能小于总需求。
例如 `34359738368` 表示整池 32 GiB，不会在 TP16 上分配为 16 份 32 GiB。
每个 EP rank 的专家分片不同，单一物理池仍需容纳这些分片之和。
加载阶段仍保留 raw CPU 专家暂存，需同时预算 CPU 暂存、最终共享池、checkpoint
加载与页缓存开销；不能只按最终权重池大小计算 host DRAM。

## DFC 对照

普通 SwiGLU 模型仍可以组合 `ascend_dfc + memfabric`。此时计算适配器另行校验
DFC 的激活、EP、workspace 等限制，见 [README_DFC.md](README_DFC.md)。
权重预取自身不套用这些激活或接收容量限制，V4 + DeepEP 也不预留 DFC workspace。

## 验证入口

- `test_dfc_model_adapter.py`：DFC 与预取的独立模型准入、V4 激活/路由保留、dense 前缀。
- `test_dfc_weight_offload.py`：原生量化缓存导出、物理布局、跨 forward 双槽保护、辅助流及异常。
- `test_moe_weight_offload_config.py`：独立参数校验、colocated/prefill/decode 和工厂组合。

设备验收应比较同一后端的全 HBM与预取输出，覆盖多 chunk、连续 decode、零 token rank、
不同层数和 stream 交接，再记录 HBM/host 峰值、SDMA 等待、TTFT 和吞吐。
