# A3 DFC Prefill 权重预取首版

本实现基于 SGLang `545ed1c664`，只修改 SGLang。
外部接口按 MemFabric `offload-layer` 的 `00b7195d` 核对：ZBAL native
`Buffer.dispatch_ffn_combine` 和 `offload.SdmaCopyPlan`。
实际运行依赖镜像内安装的 wheel，源码目录存在这些接口不代表 wheel 已更新。

当前状态：已编写接入代码与测试用例，只进行了源码静态检查。
尚未运行 CPU 模拟测试、完整编译、A3 算子精度、PD 集成或性能验证。
不得将下面的命令视为已经验证的部署结果。

DFC 的独立接入和全 HBM 用法见 [README_DFC.md](README_DFC.md)。
本变更在其上增加共享池与预取，原有 DFC 模型适配器负责通信初始化和生命周期。

## 执行行为

- `--moe-a2a-backend ascend_dfc` 选择 DFC，保留现有 router 和 shared 输出合并。
- `--moe-weight-offload-backend none` 为全 HBM 的 DFC 对照。
- `--moe-weight-offload-backend memfabric` 保留首个实际 MoE 的专家权重，后续层使用两个 HBM 槽。
- Routed W13/W2、scale、offset 和独立 shared MLP 的执行 Tensor 一起卸载。
- 在实际 MoE decoder 的入口提交下一 MoE 的搬运，MoE 模块入口等待 ready，全部 routed/shared 读取后记录 consumed。
- DeepSeek/GLM 前导 dense 层原样保留，不占用 MoE 层索引；首个实际 MoE 常驻，后续槽按实际 MoE 序号轮转。
- 预取边界通过 eager module hooks 接入，不改模型 Attention 或模型前向主体；退出时移除 hooks。
- 不重排 Attention 或 IndexCache；warmup/serving 更换计算流时通过设备事件串行交接。
- 带版本独立的事件跨 forward 保活，避免奇偶层数、连续 chunk 之间覆盖旧读者。

新后端复用 SGLang 已有 ZBAL 初始化和内存预算配置。
DFC 通信元数据设为 8192 KiB；mixed allocator 的初始化提前到 KV 池定容前。
普通后端不自动切换到 ZBAL，DFC 的独立 EP 通信组只在新后端开启时创建。

## 首版支持范围与限制

目标为 A3、本地 ModelSlim W8A8、BF16 激活、PP1、MoE TP1、固定 EP、eager。
模型路径覆盖 Qwen3-MoE、DeepSeek-V3/V3.2/R1、普通 GLM-5.3，详细 model type 见 README_DFC.md。
已撤回 V4 专用接入；DeepSeek-V4 和 GLM-5.3-Flash 因裁剪语义不受当前 DFC 支持。
模型路径适配完成不代表某个实际 checkpoint 已通过精度和性能验收。
当前 native DFC 限制 EP 不超过 16，总专家数不超过 4096 且能被 EP 整除。
共享专家融合、TBO/SBO、CP/DCP、投机解码、EPLB、冗余专家、DWDP、LoRA、memory saver、
通用 CPU offloader、CPU 权重备份、torch.compile、运行期 PD 角色切换和 remote/IPC 加载器
不能与该路径同时开启。
Decode 继续使用原有部署后端。普通 host KV 缓存可以保留，MemFabric HiCache 不能与权重池共用实体。

DFC 当前接口不能表达 routed SwiGLU clamp 或非零 expert weight offset。
遇到已配置的 `swiglu_limit`（包括显式零）、非 W8A8 scheme 或非零 offset 时会拒绝加载。
目标 checkpoint 必须先通过这些语义检查。

首版复用原生 checkpoint loader：模型构造时把后续层专家参数移到 CPU，
加载后按量化组件逐层物化、转换、导出并绑定固定槽。
这避免启动时保留完整专家 HBM，也避免另写一份 checkpoint 名称/EP 切分规则。
与方案中的完全流式 CPU 加载相比，此版本有额外 CPU 暂存：启动期应预算一份 raw CPU
专家参数加一份最终共享池，以及 checkpoint 暂存、页缓存和池 GB 对齐开销。
这版没有实现逐 checkpoint tensor 的 CPU 暂存消除。

所有层的专家执行包需要同形状、同 dtype、同 format/物理大小；不同 Attention 层不受此限制。
物理复制支持 ND 和 FRACTAL_NZ，拒绝有偏移的 storage view 和部分 ND allocation。
参数绑定共享槽后，在线权重更新、检查点导出和按名读取原始权重被拒绝。

## 启动配置

只在已获得编译/执行授权的现有镜像中运行以下命令。
本任务未安装依赖、构建替代环境或执行这些启动命令。

复制 `dfc_weight_pool.example.json`，填写本作业的池端点与 `pool_id`。
池端点必须与其他作业隔离；`pool_id` 用于进程间一致性检查，不提供额外的 native store 命名空间。
首版所有 Prefill EP rank 共同组成 SHARED 池，各 rank 物理存放自己的专家分片。
`reserve_bytes_per_rank=auto` 按实际执行 Tensor 的物理大小计算并对齐 GiB。

沿用现有 W8A8 部署的 CANN、可见设备、PD 传输和 ZBAL 环境设置，追加新后端参数。
八卡示例（两个 rendezvous 分别用于 ZBAL 和 MemFabric，需使用各自独立端点）：

```bash
export MODEL_PATH=/path/to/Qwen3-30B-A3B-W8A8
export POOL_CONFIG=/path/to/dfc_weight_pool.json
# 按现有机器的 ZBAL 内存预算设置 SGLANG_ZBAL_LOCAL_MEM_SIZE（单位 MiB）
# 并设置本 Prefill 作业的 SGLANG_ZBAL_BOOTSTRAP_URL。

bash test/manual/ascend/launch_dfc_prefill.sh "$MODEL_PATH" "$POOL_CONFIG" \
    --trust-remote-code \
    --tp-size 8 --ep-size 8 --enable-dp-attention --dp-size 8 \
    --page-size 128 \
    --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
    --mem-fraction-static 0.8 \
    --disaggregation-transfer-backend ascend
```

其余 PD 地址、端口、NIC、模型对应的 Attention 后端和 Decode 连接参数沿用已有部署。
脚本不再强制 `--attention-backend dsv4`；使用目标模型已验证的后端或平台默认解析。
新后端在配置解析中关闭图执行，无需沿用旧分支的 `--attn-dp-size` 参数。
多机场景必须维持本机 TP rank 数相等，并在启动前验证外部共享池/SDMA 的实际拓扑可达性。

DFC 接收容量默认为 `max(max_prefill_tokens, chunked_prefill_size) × top_k × EP` 的保守展开行数。
`--dfc-max-output-tokens` 可增大该容量，不能小于这个预算。
更大的 prefill 上界会增大 native workspace；需一起评估 HBM，不能直接复制 FuseEP 的容量值。
KV 定容前会按当前 native 分配公式扣除 workspace 上界，保留原有激活等运行余量。
该预算依照 `00b7195d` 中的公式和保守 A3 core 数上界计算；升级外部实现时应重新核对。
超出 source token 上界的输入会报错，而不会进行越界读取。

## 后续验证入口

所有可能触发编译的测试/启动仍需用户明确授权，并在指定镜像中使用现成完整构建流程。
源码中未添加自动构建或安装步骤。

- CPU 用例：`test/registered/unit/npu/test_dfc_model_adapter.py`、`test_dfc_weight_offload.py`。
  覆盖模型准入、dense 前缀、实际 MoE 序号、Qwen 缺少 shared/alt 属性、配置和物理布局，
  以及真实 module hooks 配合模拟复制队列的连续 forward、stream 交接和异常后禁止复用；尚未执行。
- A3 先对照全 HBM DFC 与原算子，确认 scale、gate/up、routed scaling、shared 和零 token rank。
- 再对照全 HBM DFC 与共享池预取 DFC，检查恢复后的权重、logits 和完整生成结果。
- PD 验证保持 Decode 配置一致，覆盖多 chunk、DeepSeek/GLM IndexCache 和不同 rank 的 token 数。
- 固定 KV 容量测权重 HBM 释放量，再固定 HBM 预算测可用 KV 容量与吞吐。
- 记录启动/运行峰值、CPU/DRAM 峰值、ready 等待、DFC 退化、TTFT P50/P99、KV 传输时延。

正常分布式清理在销毁模型并行组之前排空 DFC 读者和复制，销毁计划后再释放共享池。
不使用 Python finalizer 跨进程释放池；失败排空不能证明完成时保留资源并传播错误。
