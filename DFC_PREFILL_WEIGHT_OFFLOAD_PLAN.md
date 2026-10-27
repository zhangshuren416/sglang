# DFC 接入与 PD Prefill 权重预取方案（R4）

当前修订撤回原 R3 的 DSV4 专用适配，目标改为普通 SwiGLU 的 MoE 模型。
仅修改 SGLang；外部接口仍按 MemFabric `offload-layer` 的 `00b7195d` 核对。
现有两个提交保留在历史中，本次修订为工作区修改。

## 1. 状态和范围

已编写模型接入、池管理和预取代码；仅做源码静态检查。
CPU 模拟、完整构建、A3 精度、PD 集成和性能均未执行。
下列是已实现的模型构造路径，不是已验收的 checkpoint 清单。

| 模型 | model_type | SGLang decoder |
| --- | --- | --- |
| Qwen3-30B-A3B / 235B-A22B | `qwen3_moe` | `Qwen3MoeDecoderLayer` |
| DeepSeek-V3 / V3-0324 / V3.1 / R1 / R1-0528 | `deepseek_v3` | `DeepseekV2DecoderLayer` |
| DeepSeek-V3.2 | `deepseek_v32` | `DeepseekV2DecoderLayer` |
| 普通 GLM-5.3 | `glm_moe_dsa` | `DeepseekV2DecoderLayer` |

标准 DeepSeek-V4 和 GLM-5.3-Flash 的 SwiGLU clipping 无法由当前 DFC 表达，
不再宣称适配，也不删除 clipping 来改变模型语义。

每份目标 checkpoint 还必须满足：routed projections 均为 ModelSlim W8A8 INT8、
per-channel scale、零 weight offset、普通 gated SiLU，不依赖额外 alpha/beta/clamp。
`swiglu_limit` 必须缺省或为 None；不把显式 0 统一解释为关闭。

固定 EP 跨 TP、MoE TP1、PP1、BF16、eager；其余尚未适配的组合仍由配置层拦截。
这些组合限制不等价于外部 DFC 算子本身的限制。

## 2. DFC 接入

原有 Router/TopK → FusedMoE DFC 分支 → native dispatch_ffn_combine
→ 原有 routed scaling 与独立 shared expert 输出合并。

- `dfc.py` 管理 native Buffer/EP group、容量、workspace 和 forward。
- W13/W2 一次性转换为 INT8 NZ [E,H,2I]/[E,I,H]；scale 一次打包为 INT64。
- offset 非零时拒绝加载；不在热路径调用外部 FP32→CPU scale 转换。
- `dfc_model_adapter.py` 根据 model_type 和 decoder 路径验证模型，再检查 FFN 与原始权重形状。
- DeepSeek/GLM 的 dense 前缀保持原样；只将真正的 MoE decoder 加入执行索引。
- Qwen3 的 MoE 选择 EP forward，避免误入 TP 路径追加 all-reduce；保留原 TopK padding 元信息。
- routed 几何要求一致，以便共享一个容量预算；模型名和 W8A8 标签本身不足以证明兼容。
- ZBAL 通信元数据为 8192 KiB，mixed allocator 在 KV 定容前初始化。
- native workspace 上界从 KV 预算扣除，同时保留原有激活等 headroom。

全 HBM DFC 不依赖权重池，使用独立 DfcModelAdapter 生命周期。
驻留路径先做精度对照，再与开启卸载的同一 DFC 对照。

## 3. 权重池与加载

启用 `--moe-weight-offload-backend memfabric` 后，DfcOffloader 扩展驻留适配器。

1. 原生 make_layers 逐层构造；保留完整 decoder 列表和 dense 层顺序。
2. 首个实际 MoE 的 routed/shared 权重留在 HBM，其余 MoE 的目标参数移到 CPU。
3. 原生 loader 完成名称映射及 TP/EP 切分。
4. 加载后逐量化组件回到 NPU，进行一次执行格式转换。
5. 导出 NZ/ND 的物理字节到 SHARED DRAM；按真实 storage 大小检查和分配。
6. 为后续层建立固定 SdmaCopyPlan，目的地址为两个固定 HBM 槽。
7. 各层参数绑定到对应槽，正常 forward 无格式转换或 host synchronize。

运行期专家 HBM 约为 M0 加两个完整 MoE 槽。Router、Attention、norm、embedding
等保持原有路径。DeepSeek/GLM 的独立 shared MLP 同属该层权重包；Qwen3 没有 shared 分支。

首版保留一份 raw CPU 专家参数暂存，另有最终 DRAM 执行包、checkpoint 暂存和页缓存。
并未实现完全流式 CPU 加载，也未测得实际启动峰值。

池使用每个 TP/EP rank 自存专家分片的 SHARED 布局。容量按物理字节计算并对齐 GiB，
rank 间检查配置和布局一致。每个作业使用独立 store 端点；pool_id 不提供 native 命名空间。
HiCache 的 MemFabric offload 实体不能与当前权重池同时初始化。

## 4. 预取边界

采用受支持 decoder 和 MoE 模块的 eager forward hooks，不再修改模型前向主体：

- decoder pre-hook：进入实际 MoE decoder 时，提交下一个实际 MoE 的预取。
- MoE pre-hook：等待当前层 ready，及下一层短 AIV submit 的 submitted 事件。
- MoE post-hook：全部 routed/shared 读取结束后记录 consumed，推进 MoE 序号。
- decoder always-call post-hook：Attention 在 MoE 前失败时也标记执行失败。
- MoE always-call post-hook：MoE 抛错时关闭 scope，禁止继续复用槽。

submitted 表示短提交任务完成，ready 表示 DMA 完成；两者不能互换。
复制流覆盖某槽前，必须等待该槽上一个读者的 consumed。
每轮使用独立事件，跨 forward 保活；warmup 和 serving 切流通过设备依赖串行交接。
Qwen3 不要求存在 DeepSeek 的 shared/alt_stream 属性。

dense decoder 没有这些 hooks，不占用 MoE 索引。绝对 decoder layer_id 可以从 3 起步或不连续，
但槽和计划始终按实际 MoE 序号分配；M0 对应首个实际 MoE，而不是模型第 0 层。

## 5. 配置与文件

- `--moe-a2a-backend ascend_dfc`：选择 DFC。
- `--moe-weight-offload-backend none`：全 HBM DFC 对照。
- `--moe-weight-offload-backend memfabric`：共享池及 SDMA 权重预取。
- `--moe-weight-offload-config <JSON>`：池配置。
- `--dfc-max-output-tokens`：保守展开容量，可增大默认值。

示例和启动方法见 `test/manual/ascend/README_DFC.md` 与
`README_DFC_WEIGHT_OFFLOAD.md`。脚本为 `launch_dfc_prefill.sh`，不再强制 dsv4 Attention 后端。
目标模型的 Attention、PD 地址、端口和 NIC 配置沿用各自现有部署。

V4 模型文件已恢复至本任务开始前基线，不再包含 DFC 专用 hook 或加载特例。
已有公共 DFC 后端、NPU 量化转换和资源生命周期保留。

## 6. 验证与交付边界

新增/更新的 CPU 用例（尚未执行）：

- `test_dfc_model_adapter.py`：dense 前缀、Qwen 属性差异、模型准入、clipping、几何一致性及卸载组件收集。
- `test_dfc_weight_offload.py`：物理布局、容量、真实 module hooks 配合可控复制队列，
  非连续 layer_id、1–5 层、连续 forward、切流及 Attention/MoE 异常。

设备验收仍必须包含：

1. 原模型 vs 全 HBM DFC：scale、gate/up、routed scaling、shared、padding、零 token rank 和完整输出精度。
2. 全 HBM DFC vs 卸载 DFC：恢复字节、logits 和生成结果，覆盖多 chunk/不同 rank token 数。
3. dense 前缀后首个实际 MoE 常驻、两槽轮转及持续请求的覆盖保护。
4. 固定 KV 容量测权重 HBM，再固定 HBM 预算测 KV 容量、TTFT、吞吐和 PD 传输影响。
5. 启动/运行/CPU 峰值、ready 等待及退出时的资源排空。

清理依次排空消费者与复制、移除 hooks、关闭计划、释放池、关闭 DFC 资源，
然后销毁原模型并行组；不使用跨进程 Python finalizer 释放池。
