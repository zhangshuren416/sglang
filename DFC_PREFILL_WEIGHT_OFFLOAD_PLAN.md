# PD Prefill 的 DFC 替换与 MoE 权重预取实现方案

状态：修订 R3，已根据用户“看下 memfabric 的接口然后动手做一版”实施首版源码。
代码和验证说明见 `test/manual/ascend/README_DFC_WEIGHT_OFFLOAD.md`。
目前仅完成静态检查，CPU 模拟、完整编译、A3 精度、PD 集成和性能均未执行。

日期：2026-10-27。

基线：SGLang `codex/ascend-release-2026930-dev`，HEAD
`545ed1c6649321aa61b57a287032a7791690247e`。

本版替代上一版基于旧分支的方案。以下类、方法、配置和文件清单按本基线重新核对；
不能沿用上一版的平台拆分文件或启动脚本。实施各阶段开始前再次读取 branch/HEAD/status，
如果用户又切换分支，先重新核对受影响的接入点，不在未核对的新基线上套用计划。

范围：仅修改 SGLang。ZBAL DFC、MemFabric shared DRAM 和 `SdmaCopyPlan` 是外部依赖，
本计划不修改这些项目。外部接口缺失或目标配置不兼容时，报告具体阻塞，重新评审受影响部分。

R3 对照的外部接口为 MemFabric `offload-layer` 的 `00b7195d`：
native `zbal.zbal.deepep_adaptor.Buffer.dispatch_ffn_combine` 返回 output、计数和 workspace；
`offload.SdmaCopyPlan.enqueue(submitted_event)` 在 submit 与 Notify wait 之间记录 submitted。
SGLang 直接适配这些接口；不调用高层 Buffer 的每次 CPU scale 打包，也不接管外部整模型循环。
首版复用原生 loader 并暂存 CPU 专家参数，没有实现原 R2 的 meta skeleton 和完全流式 CPU 加载；
这项差异及 CPU 峰值预算在第 6 节明确记录。

## 1. 目标和首版边界

目标调用链：

```text
SGLang 的原有路由
  → 等待当前 MoE 权重 ready
  → ZBAL Buffer.dispatch_ffn_combine
  → 原有 routed/shared 输出合并

独立预取流：MemFabric shared DRAM → 下一层的固定 HBM 权重槽
```

首版完成以下能力：

1. 在 PD 的 Prefill 实例中，用 DFC 替换当前的 `fused_deep_moe` 调用。
2. 首个实际 MoE 层的专家权重常驻 HBM；其余层预存于共享 DRAM。
3. 使用两个固定 HBM 槽，计算当前层时搬运下一层。
4. 逐层加载和转换专家权重，启动阶段也不要求完整 MoE 权重驻留 HBM。
5. 支持多次请求、chunked prefill、EP rank 输入不均衡，以及本地零 token 的参与 rank。
6. 通过外部能力检测、数值对照、内存测量和设备时间线证明实现成立。

首版目标配置：Ascend A3、DeepSeek-V4 的本地 ModelSlim W8A8 checkpoint、BF16 激活，
`PP=1`、`MoE TP=1`、固定 EP，优先验证 TP8/EP8/attention-DP8 场景。
本分支以 `--tp-size 8 --ep-size 8 --enable-dp-attention --dp-size 8` 表达该并行布局；
`attn_dp_size` 是解析后的派生值，不使用旧基线的 `--attn-dp-size` 参数。
实际 checkpoint 的激活函数、量化参数及布局是否被 DFC 支持，是第一个实施阶段的验收条件。

首版运行 eager，使用单条模型计算序列和一条权重预取流。
关闭 TBO/SBO、prefill 图执行、投机解码、EPLB、冗余专家及 elastic EP。
CPU 可以提前提交后续 batch，但两个模型 forward 不在设备上交错访问权重槽。
Decode 实例按原配置工作，不开启此权重卸载。

后续单独评审：其他模型/量化、异形状专家层、PP/CP、图执行、交错 microbatch、
专家选择性预取、更深预取、多副本权重去重和运行期权重更新。

### 1.1 权重范围

| 数据 | 首个 MoE 层 | 后续 MoE 层 |
| --- | --- | --- |
| Routed expert 的 W13/W2、scale、执行必需的量化数据 | HBM 常驻 | DRAM + 两个 HBM 槽 |
| 独立 shared expert 的矩阵、scale、执行必需的量化数据 | HBM 常驻 | 纳入同层权重包，同步预取 |
| Router/gate、路由 correction bias、norm | HBM 常驻 | HBM 常驻 |
| Attention、embedding、lm_head | 原有路径 | 原有路径 |

这里卸载的是专家执行数据；router 保留，用于保持原有路由和准备路径。
shared expert 不因关闭融合而全部残留在 HBM 中。

## 2. 已核实的 SGLang 接入点

| 文件 | 当前事实 | 计划 |
| --- | --- | --- |
| `hardware_backend/npu/moe/fuseep.py` | 调用 `buf.fused_deep_moe`，包含 FuseEP 专用后处理 | 保留原路径，新建 DFC 适配 |
| `layers/moe/fused_moe_triton/layer.py` | `FusedMoE.forward` 有 FuseEP 直接调用分支 | 增加 DFC 分支及 StandardDispatcher 占位 |
| `layers/moe/utils.py` | 后端枚举、共享专家配置、MoE 运行状态 | 增加 `ascend_dfc`，补齐对应分类 |
| `hardware_backend/npu/quantization/moe_methods.py` | W8A8 后处理根据 FuseEP 后端转换布局 | 加入独立的 DFC 一次性后处理 |
| `models/deepseek_v2.py` | DSV4 复用此处的 MoE、shared expert 和路由逻辑 | 补齐 DFC 的 A2A/共享专家执行语义 |
| `models/deepseek_v4.py` | `DeepseekV4DecoderLayer`、NPU MoE 同步、模型 forward 和原生 loader 均在此文件 | 在现有类中接入受控逐层加载、层及 forward 生命周期 |
| `utils/common.py` | V4 使用 `make_layers` 构造层；可传 offloader kwargs，当前 V4 未传 | 不假设存在旧基线的 layer-stack/平台拆分接入点 |
| `utils/offloader.py` | V2 在当前模块 forward 结束后提交后续搬运 | 不直接复用该调度；新路径与 V1/V2 互斥 |
| `model_loader/loader.py` | 现有 layered loader 要求 `load_weights_to_module` | DSV4 不能直接假设可用，需要专用加载适配 |
| `mem_cache/pool_host/npu_memfabric.py` | HiCache 使用 LOCAL 池和首次初始化容量 | 首版不复用该单例；拒绝同时占用同一外部 offload 实体 |
| `arg_groups/fields/parallel.py`、`parallel_hook.py` | 输入为 `enable_dp_attention`/`dp_size`，attention 宽度按本分支规则派生 | 新后端兼容检查使用当前规则和参数 |
| `hardware_backend/npu/attention/ascend_dsv4_backend.py` | 本 HEAD 包含跨层 IndexCache top-k 复用 | 保持其 producer/consumer 顺序；加入功能及覆盖窗口回归 |

以上路径均相对于 `python/sglang/srt/`。

当前 V4 的具体接入点：`DeepseekV4DecoderLayer.forward`（约 3010 行）、
`forward_hc_pre_from_prev`（约 3508 行）、`_run_moe_ffn_dp_sync`（约 3752 行）；
`DeepseekV4Model._forward_layers_hc_pre_from_prev`（约 4384 行）、`forward`（约 4675 行）；
`DeepseekV4ForCausalLM.post_load_weights`（约 5200 行）、`load_weights`（约 5360 行）。
行号仅用于本 HEAD 定位，实施以方法名称和实际代码为准。

## 3. 后端和配置设计

### 3.1 单独增加 `ascend_dfc`

新增 `--moe-a2a-backend ascend_dfc`，由 `FusedMoE.forward` 直接调用 DFC。
原有 `ascend_fuseep` 继续调用原算子，便于精度和性能对照。
新路径直接导入 ZBAL，不通过 `SGLANG_ZBAL_LOCAL_MEM_SIZE` 的正值猜测应加载哪个 Buffer。

新增后端必须同时审计和补齐：

- CLI 的 Literal/choices、后端枚举、EP 解析及运行状态初始化；同时审计本分支
  `parallel_hook.py` 的后端白名单以及 spec 字段中的后端列表。
- MoE A2A 标志、shared expert 的 TP1 选择、原生路由分支。
- StandardDispatcher 占位，避免再次进行普通 dispatch/combine。
- 路由 top-k 输出格式，保证 DFC 获得真实的 ID 和 routing weights。
- 量化后处理、模型 override、DP 准备及后端兼容性判断。

首版关闭 shared expert fusion，通过配置解析流程声明；shared expert 走原有独立 MLP 分支。
这样 DFC 只接收 routed experts 的编号，不引入当前 DFC 未验证的每 rank 复制 shared 槽语义。

### 3.2 建议新增的配置

以下接口已加入源码；实际运行仍待镜像和 A3 验证。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `--moe-a2a-backend ascend_dfc` | 原默认不变 | 选择 DFC |
| `--moe-weight-offload-backend {none,memfabric}` | `none` | `none` 为全 HBM DFC 对照 |
| `--moe-weight-offload-config <JSON 文件>` | 空 | 共享池端点、作业标识、容量和传输参数 |
| `--dfc-max-output-tokens <整数>` | 自动 | DFC 最坏路由展开容量，只能扩大默认预算 |

`exec.moe` 保存 DFC 后端选择，`exec.offload` 保存卸载配置。
业务代码读取 `get_exec()` / `get_disagg()` / `get_parallel()` 的已解析配置。
初版固定首个 MoE 常驻、两个槽和提前一个 MoE 层，暂不增加没有验证用途的调优开关。

卸载配置 JSON 的建议结构：

```json
{
  "schema_version": 1,
  "pool_id": "prefill-job-unique-id",
  "store_url": "tcp://POOL_STORE_HOST:PORT",
  "reserve_bytes_per_rank": "auto",
  "sdma_channels": 4
}
```

`reserve_bytes_per_rank=auto` 按实际执行格式的权重清单计算，并按外部池的要求对齐。
如果提供整数，必须覆盖需求。所有 rank 使用相同的池级 reserve，物理贡献可以不同。
pool rank、global rank、EP rank、device ID 的映射由运行时显式生成并记录。
首版使用计算进程共同建立的 embedded SHARED 池，不新增独立 provider 服务。

### 3.3 启动校验

配置解析阶段校验角色、并行、图执行、互斥 offloader 和动态专家功能；加载阶段校验外部能力、
checkpoint、dtype、专家数、真实布局及源池容量。错误包含不兼容的具体字段。

启用卸载要求 `disaggregation_mode=prefill`；全 HBM DFC 可用于离线对照测试。
不允许在 warmup 或非 EXTEND 的必要内部调用中绕过权重准备。
若 Prefill 实例意外执行不受支持的模式，应在消费槽位之前明确报错。

首版拒绝同时使用 MemFabric HiCache host backend，避免两个组件各自初始化、定容或销毁同一外部实体。
其他不占用该实体的缓存路径，按实际依赖及带宽验证结果决定是否允许。

## 4. DFC 适配：先完成全 HBM 路径

新增 `hardware_backend/npu/moe/dfc.py`，提供：

```python
def forward_dfc(layer, hidden_states, topk_output):
    # 权重由 resident 参数或当前槽的固定绑定提供。
    output, expert_token_nums = runtime.buffer.dispatch_ffn_combine(
        hidden_states,
        topk_idx=topk_output.topk_ids,
        weight1=layer.w13_weight,
        scale1=layer.w13_weight_scale,
        weight2=layer.w2_weight,
        scale2=layer.w2_weight_scale,
        topk_weights=topk_output.topk_weights,
        max_output_size=runtime.capacity,
        num_experts=runtime.num_routed_experts,
        quant_mode=1,
    )
    return output
```

该代码表示职责和参数映射，正式实现以外部版本的签名为准。

### 4.1 通信和容量

`DfcRuntime` 持有 EP group、ZBAL Buffer、容量和必要 workspace 预算，挂在 RuntimeContext resources。
进程组成员按真实 global rank 建立；不能用 EP rank 直接充当本机 NPU device ID。
通过 SGLang 的窄适配器创建 `backend="zbal"` 的独立 EP 进程组，初始化并取得其通信元数据，
再构造外部已有的 native Buffer binding。适配器暴露上述 `dispatch_ffn_combine` 调用形状，
规范化 native 返回值与临时 workspace 生命周期，不依赖高层 Buffer 隐式克隆默认通信组。
实际 native 构造签名和 backend 能力在 S0 检测，元数据初始化失败直接退出。
不把整个 SGLang 默认通信后端替换成 ZBAL，也不修改普通 Attention/PD 通信组。
临时 Tensor 遵守当前流/`record_stream` 的 allocator 生命周期，不把每层 workspace 累积保留到整个 forward 结束。

Buffer 仅初始化一次，初始化及 group 创建顺序在所有参与 rank 一致。
容量按 DFC 的接收行语义推导；采用全组 source token 上界和 top-k 展开的保守预算，
验证最偏斜路由下不越界。不能直接沿用 FuseEP 同名参数的数值或默认值。
同时预留通信窗、workspace 和最坏 shape 的运行内存，再进行 KV 池定容。

### 4.2 数值兼容条件

- 原有 router、HashTopK/TopK、correction bias 和专家编号保持数学语义。
- top-k ID 使用外部接口要求的整型，routing weights 使用 FP32。
- `routed_scaling_factor` 只作用一次，shared 输出不误乘 routed scale。
- DFC 的 W1/W2 与 gate/up 排列按数值对照确认，不盲目沿用 FuseEP mode 1/2 转换。
- scale 在初始化时打包为 INT64，热路径不触发外部 Python 包装的 FP32 → CPU → INT64 转换。
- 明确核对激活函数、`swiglu_limit`、非零 offset、bias、平滑量化等参数。

如果 checkpoint 依赖 DFC 接口无法表达的语义，当前配置视为不兼容。
在仅修改 SGLang 的范围内，不能静默忽略该语义，也不能用同名算子来推定数学等价。
这是开始完整模型集成前必须报告的兼容性结果。

## 5. 权重包、执行格式和共享 DRAM

`hardware_backend/npu/moe/dfc.py` 实现格式转换和 workspace 预算；
`dfc_model_adapter.py` 实现独立的 DFC 模型接入和生命周期；`dfc_config.py` 实现池配置和物理布局检查；
`dfc_offload.py` 实现逐层执行包清单、槽位绑定和导出。

每个 `LayerWeightManifest` 至少包含：

```text
模型标识、checkpoint/量化配置指纹、layout_version
全局层号、MoE 顺序号、EP 分片及专家范围
每个 Tensor 的模块路径、dtype、逻辑 shape、stride、NPU format
实际物理 storage 字节数、源池 rank/GVA/偏移、目标槽内偏移
shared expert 的对应数据、包的总容量及初始化校验结果
```

第一个 MoE 的完整包保留独立 HBM storage。
后续层按 `slot=(moe_index-1)%2` 固定绑定到 A/B。
两个槽的每个 Tensor 是按执行格式实际分配的 Tensor，首版要求专家包 geometry/layout 兼容。
同一个 storage 可以由多个层的 Parameter 持有引用，但物理权重容量始终只有两个槽。
绑定仅在初始化时完成；forward 不逐层创建新 Parameter 或修改 `.data`。

### 5.1 NZ 权重导出与恢复

1. 在单层 scratch 中生成 DFC 的正确执行 Tensor。
2. 获取并验证其物理 storage 大小、起始地址、format、padding 和合法范围。
3. 通过外部已有的原始地址复制能力，将物理字节写入本 rank 拥有的共享 DRAM。
   启动阶段可使用映射 DRAM 上的 sparse-copy；显式处理需要 DVA 的分配模式。
4. 使用固定 `SdmaCopyPlan` 从该 DRAM 恢复到真实 NZ 槽，检查字节和 DFC 输出。
5. 通过后才发布该层清单并释放 scratch。

SGLang 层的格式 helper 封装这些检查。禁止用普通 Tensor `.cpu()`/`.copy_()`
来假定已经保存或恢复 NZ 物理排列；禁止只用逻辑 `numel * itemsize` 推断物理容量。
首版采用整块、独立 owning storage，避免复杂切片的 format/offset 推断。
若目标 torch_npu 无法可靠提供所需 storage 信息，在能力验证阶段报告阻塞，不伪造 NZ Tensor。

shared MLP 使用自身量化方法要求的格式；它与 routed DFC 的格式可以不同。
同一个层包中的全部 Tensor 完成后，才记录该层 `ready`。

### 5.2 共享池和外部能力检查

新增 `MemfabricWeightPool` 适配器，使用已有外部 Python API：初始化 SHARED 池、分配、
映射与原始地址复制、准备及提交 `SdmaCopyPlan`、关闭计划和池。
其使用只发生在新功能开启时，普通启动不强制导入这些依赖。

默认每个 EP rank 把自己的分片存放在本 rank 的 DRAM，避免 rank 0 集中加载和热点。
所有贡献片段属于同一个 SHARED 池；通过独立测试让其他 rank 消费指定片段，证明共享访问可用。
物理容量按各 rank 的贡献求和；reserve 的虚拟跨度不能重复记作物理内存。

启动日志记录外部包版本、实际 import 路径和能力检测结果。
代码目录中有新接口，不代表运行时 wheel 已提供对应能力。
检查 SHARED、`SdmaCopyPlan`、submitted/ready 语义、通道约束和实际源地址可达性。
外部只允许一条复制流时，所有权重计划固定使用该流；通道数在整个池生命周期内固定。

## 6. 逐层加载：把启动 HBM 峰值纳入实现

首版通过现有 `make_layers` 的 offloader 扩展和 `DefaultModelLoader.postprocess_weights`
加入组件后处理，不新增独立 checkpoint loader。不兼容的远端、IPC、在线更新等路径明确拒绝。

1. 每个 decoder 构造完成后，立即将 M0 以外的专家 Tensor 放到 CPU，保留 Parameter 子类和 loader 属性。
2. 原生 DSV4 `load_weights` 完成名称映射、TP/EP 切分及其线程池任务；检查未加载参数。
3. 依原生模块顺序，每次只将当前 routed/shared 量化组件放回 NPU，并执行一次后处理。
4. 以首个非驻留层的实际物理布局建立两个槽，校验其他层一致；导出 raw NZ/ND 字节到 SHARED 池。
5. 导出完成后，将执行 Tensor 绑定到固定槽并释放转换源；初始化阶段允许同步，运行期不做格式转换。
6. 建立各层固定 SDMA 计划、DFC 通信组和元数据；从 KV 定容预算扣除 native DFC workspace 上界。

该方案实现启动期专家 HBM 不随层数线性累计，但与 R2 的完全流式 CPU 方案存在差异：
启动期 CPU 仍需要一份 raw 专家参数，最终共享 DRAM 需要另一份物理执行包，另加 checkpoint
暂存、page cache 和 GiB 对齐。首版没有消除完整 raw CPU 暂存；物理峰值需要实机测量。
所有导出在原生加载结束后进行，不与 loader future 并发复用 Tensor。
不对运行槽 backing storage 调用破坏性 `resize_(0)`。

## 7. 双缓冲执行时序

新增 `hardware_backend/npu/moe/dfc_offload.py`，提供 pool facade、scheduler 和 forward generation。
核心接口建议为：

```python
begin_forward(forward_batch)       # 分配本次 generation；继承前次各槽最后使用依赖
on_layer_enter(layer_id)           # 尽早提交下一 MoE 的预取
before_moe(layer_id)               # 等当前权重 ready，验证层号/版本/槽位
after_moe(layer_id)                # routed/shared 全部读完后记录 consumed
end_forward()                     # 记录完成；保留异步资源直到完成
close()                           # 排空，销毁计划，释放池
```

运行状态由 runner/model 实例持有，或通过 RuntimeContext resources 注册。
forward epoch 和槽位状态属于该调度器，不放入新的模块级全局变量。
预取流使用独立资源名 `moe_weight_prefetch`，不复用 attention 的 alt stream 或 PD KV 传输流。

### 7.1 首次 forward

```text
计算流：  M0（常驻）        M1(A)             M2(B)             M3(A)
预取流：  M1→A             M2→B              M3→A              M4→B
写入前提：A 首次空闲        B 首次空闲        等 M1 consumed     等 M2 consumed
读取前提：M0 已初始化       等 M1 ready      等 M2 ready        等 M3 ready
```

在当前 Transformer 层开始时，尽早安排下一个 MoE 的搬运。
在当前层进入 MoE 之前才等待其权重，Attention 的计算可以继续覆盖搬运。
模型中若存在非 MoE 层，使用实际 MoE 顺序表，不能按 `layer_id+1` 猜下一个 MoE。

### 7.2 Event 协议

每次槽使用具有 `(model_version, forward_epoch, layer_id, slot, generation)` 身份。

```text
预取流：wait(目标槽上次 consumed)
       → plan.enqueue(submitted_event)
       → 外部 SDMA Notify wait
       → record(ready)

计算流：Attention 等当前层前置计算
       → wait(当前层 ready)
       → wait(下一层 submitted，存在时)
       → 当前层 routed DFC + shared MLP
       → join 所有权重读取流
       → record(consumed)
```

`submitted` 用来让短暂的 AIV 提交工作先完成，再进入可能占用大量 AIV 的 MoE 计算。
它不证明数据已搬完；ready 必须位于外部复制完成等待之后。
一个 plan 可能包含多段 W13/W2/shared/scale，不能首段结束即发布 ready。

首次空槽不等待未记录的 consumed。复用时等待目标槽的上一使用者，不能等待当前层读完。
M0 常驻数据不需要逐次搬运；其首次初始化完成也有明确依赖。
Python 的状态只表示任务已安排；数据是否完成由设备事件证明。

### 7.3 连续 forward 和 chunk

下一次 forward 的 M1 不一定拥有空槽：它要等待上一次 forward 对 A 的最后读取。
槽位依赖跨 forward 继承，不能在 `begin_forward` 无条件清空。
奇数/偶数 MoE 层数量及小于三层的情况都单独覆盖。
每个 chunk 视为一次新的 forward generation，执行相同协议。

旧 generation 的 Event、Tensor、计划和映射保持存活，直到其设备任务完成。
事件只有在旧 generation 的所有 wait/record 完成后才能复用；不逐层反复覆写同一个 Event。
使用完成查询回收 generation，容量与引擎允许的在途 forward 预算一致。
超预算时使用明确的入队背压，不释放或复用仍在飞的事件。

### 7.4 NPU 层接入位置

- 在 `DeepseekV4Model.forward` 的完整生命周期进行 begin/end；本分支没有旧方案中的
  `NonCudaDecoderMixin`/`deepseek_v4_platform_stage.py` 接入结构。
- 常规循环与 `_forward_layers_hc_pre_from_prev` 循环均在调用当前 decoder 之前执行 layer-enter，
  避免只覆盖一条路径。其余后端在开关关闭时不进入新逻辑。
- 在 `DeepseekV4DecoderLayer._run_moe_ffn_dp_sync` 的入口进行 before，返回前进行 after。
  本方法直接管理 CP/TP/DP gather/scatter 与局部 shared 计算，必须保留现有的数据分布和相加语义；
  不把旧分支的 row-transport 抽象迁入本分支。
- ready 位于任何 shared 权重读取之前，consumed 位于全部 routed/shared 读取及必要跨流 join 之后。
  不能只包住内部 `self.mlp(...)` 或 DFC 调用而遗漏 shared 的前置计算路径。
- 本地 token 为零但 EP 仍有工作时，照常准备本 rank 专家并参与 DFC。
- warmup、profiling、启动验证使用同一调度器；不为它们临时保留全模型专家 HBM。

### 7.5 本分支的 IndexCache 与 late-layer tail

IndexCache 由当前 NPU attention backend 根据模型配置 `use_index_cache` 及 c4 层启用。
其 top-k producer/consumer 与 MoE 权重缓存属于不同数据流，不让权重 scheduler 重排 Attention 层，
也不更改跨层 top-k buffer 的复用和更新规则。
IndexCache 的跳算可能缩短前一层的计算覆盖窗口；性能报告按实际启用情况单独记录。

`_forward_layers_hc_pre_from_prev` 还管理 late-layer tail 的 token 子集切换。
权重就绪依赖按实际 MoE 层号保持，token 子集变化不减少该层完整本地专家包的预取。
若目标 NPU 配置触发该路径，验证 enter/exit 和 token 恢复，不能借用旧分支的 replay-tail 实现。

## 8. HBM、DRAM 和带宽预算

设 `Wi` 为每张卡第 i 层 routed + shared + 量化数据的实际物理执行容量。

```text
运行期专家 HBM ≈ W0 + 2 × Wslot
运行峰值 HBM  = 专家槽 + 常驻参数 + KV + 激活 + 通信窗 + workspace + 碎片余量
启动峰值 HBM  = 当时常驻参数 + M0 + 当时已分配槽 + 单层转换 scratch + 通信及临时内存
本 rank DRAM  = Σ(i>0) Wi + 池对齐/元数据 + raw CPU 专家暂存 + checkpoint 暂存
每 forward 搬运量 = Σ(i>0) Wi
```

只保留两个槽不会降低每次完整遍历所需的搬运字节数。
每层加载时间包含传输、提交和必要格式恢复；首版目标为热路径无格式转换。
是否隐藏，按实际提交到目标 MoE 消费之间的窗口判断。
通道数更多也不能突破源端 DRAM、互联、目标 HBM 或 PD/EP 共享资源的带宽上限。

内存对照分两轮：固定 KV token 容量测 HBM 释放量；固定 HBM 预算测新增可用 KV 容量。
避免 KV 自动扩容把权重节省全部用掉后，误判卸载没有收益。
采集框架 allocated/reserved 和设备实际可用内存，后者覆盖外部 runtime 分配。

## 9. 失败、退出和兼容性

| 情况 | SGLang 行为 |
| --- | --- |
| 缺少 DFC、共享池或 SDMA API | 启动失败，列出缺失能力与实际 import 路径 |
| NZ/shape/量化语义不兼容 | 在加载或预检阶段失败，不送入内核 |
| 当前槽与层号、版本或 generation 不匹配 | 在消费前报错，禁止计算 |
| 异步搬运失败或超时 | 标记执行失败，停止后续消费和槽位复用，触发引擎错误处理 |
| 一个 EP rank 失败 | 终止受影响 EP 执行，避免其他 rank 无期限等待；复用现有 watchdog/退出机制 |
| 请求取消 | GPU 已提交的访问仍排空；不按 HTTP 请求取消立即回收共享槽 |
| 运行期 reload、保存完整 state_dict、内存迁移 | 首版启用卸载时明确拒绝；槽别名不代表各层原始权重 |
| 池退出 | 所有消费者完成、计划成功排空后才卸载映射和释放池 |

退出按“停止入队 → 排空消费者/预取 → 销毁计划 → 释放绑定/映射 → 销毁池和通信资源”排序。
不能仅同步当前进程，就释放仍被其他消费端访问的共享池贡献。
失败排空保留外部接口要求保留的资源，不提前清空 owner 引用。

运行中失败不自动切回完整 HBM 模型；当前 HBM 预算可能容纳不了该模型。
回退通过停服务并使用原配置重启完成。

## 10. SGLang 文件改动清单

| 文件或模块 | 改动 |
| --- | --- |
| `arg_groups/fields/exec_.py`、`field_order.py` | 新后端及卸载配置声明 |
| `arg_groups/dfc_hook.py`（新增）、`pipeline.py`、`resolution_hooks.py`、`overrides.py` | 角色、当前分支 DP/EP、共享专家融合及功能互斥解析 |
| `layers/moe/utils.py`、`fused_moe_triton/layer.py` | DFC 分类、直接执行、dispatcher 选择 |
| `hardware_backend/npu/moe/dfc.py`、`dfc_model_adapter.py`（新增） | DFC Buffer、workspace 预算、模型接入和生命周期 |
| `hardware_backend/npu/moe/dfc_config.py`（新增） | 池配置及物理布局检查 |
| `hardware_backend/npu/moe/dfc_offload.py`（新增） | 外部池 facade、固定计划、预取和 generation 调度 |
| `hardware_backend/npu/quantization/moe_methods.py`、`npu/utils.py` | DFC 一次性转换、ZBAL 元数据和 mixed allocator 初始化顺序 |
| `models/deepseek_v2.py` | DFC 的 routed/shared 语义和所有后端相关分支 |
| `models/deepseek_v4.py` | 当前 `DeepseekV4DecoderLayer` 的 MoE 入口、两条层执行路径及原生加载窄接口 |
| `utils/offloader.py`、`model_loader/loader.py` | 原生加载后的逐组件处理和有界专家 HBM |
| `mem_cache/kv_cache_configurator.py` | KV 定容前保留 native workspace 预算 |
| `model_executor/model_runner_components/weight_{updater,exporter}.py`、`distributed/parallel_state.py` | 阻止更新/导出槽别名，模型并行组销毁前排空释放 |
| `test/registered/unit/npu/test_dfc_weight_offload.py`、`test/manual/ascend/…` | 已写 CPU 模拟测试和手工启动说明，尚未执行 |

无 MemFabric/ZBAL 源码改动；不修改当前 Decode 部署脚本的行为。
现有 offloader、FuseEP 和非 NPU 路径不因新功能被动导入依赖或改变默认值。

## 11. 实施顺序和每阶段交付

用户已授权首版实施。下表保留验证阶段与验收目标；源码落地不等同于阶段验收完成。
若兼容性检查证明需要改变算子数学语义、修改外部项目或扩大约定范围，先报告并重新评审。

| 阶段 | 工作 | 交付与通过条件 |
| --- | --- | --- |
| S0：兼容预检 | 记录真实外部版本、硬件、量化配置和形状；验证 DFC、共享池和 NZ 恢复能力 | 完整能力表；明确 activation/clamp/offset、容量及格式是否可支持 |
| S1：全 HBM DFC | 新后端、Buffer、路由和量化适配、独立 shared 分支 | 单层参考及完整模型精度；原 FuseEP 回归；得到全 HBM DFC 基线 |
| S2：权重包和池 | manifest、原始字节导出、SHARED 池、固定 SDMA 计划 | 原始数据恢复及 DFC 输出对照；跨 rank 读取；异常/关闭正确 |
| S3：逐层加载 | CPU 暂存、原生映射复用、逐组件后处理、槽位绑定 | 启动专家 HBM 不随层数线性累积；加载完整性和后处理恰好一次；记录 CPU 额外开销 |
| S4：双缓冲执行 | 完整 layer/forward hooks、三种 Event、跨 batch/chunk 依赖 | 延迟复制压力下无旧层权重、覆盖、死锁或在途资源释放 |
| S5：PD 验收 | Prefill 启用、Decode 原配置，真实请求和 KV 传输并发 | 精度、内存、吞吐、TTFT、P99、权重等待及设备重叠报告 |

每阶段提供可复现实验命令、结果和限制。单层合成用例通过不等同于完整 PD 模型通过。
若没有 A3 实机、目标 checkpoint 或可用外部 wheel，完成本地可验证部分并标明未验证项，
不把 CPU 测试或源码审查写成实机结果。

## 12. 验证矩阵和验收标准

### 12.1 数值验证

两组独立对照，区分算子替换误差和卸载错误：

1. DFC 与显式反量化/GEMM/SwiGLU/路由合并参考、原算子对照：采用与 W8A8 数值路径相适应的容差。
   容差由 S0/S1 的独立参考和现有精度规范确定并记录，不能为隐藏错误而放宽。
2. 相同 DFC、相同输入/路由：全 HBM与 DRAM→HBM 恢复比较权重字节和输出。
   权重必须逐字节一致；若相同 DFC 本身确定性成立，则输出也要求一致。
   否则以重复执行的固有波动为依据，单独报告，不能混入传输误差。

完整模型检查 logits/logprob、固定样例及现有模型精度集，重点覆盖 routed scale、HashTopK、
shared 分支、多个 chunk、前缀命中后的短 extend，以及 EP rank 无本地 token。
新增本分支的 IndexCache producer/skip 层组合回归；以相同配置的常驻 DFC 为对照，
支持关闭时再测关闭组合，不为测试强行改变 checkpoint 的既定数学语义。

### 12.2 CPU 测试

在新模块的生产逻辑边界验证：配置互斥、rank 映射、manifest 溢出/容量/格式校验、
一/二/三/多层排程、奇偶轮转、跨 epoch 依赖和失败后的禁止复用。
使用可控的异步复制依赖模拟器观察实际读取值及覆盖错误，避免只断言 mock 被调用。

按 `write-sglang-test`、`test/README.md` 和 unit-test admission 规范，
使用 `CustomTestCase`，按 source 目录归属注册测试，逐文件运行。

### 12.3 A3 测试

- 同 rank 与跨 rank 源片段、非对齐尾块、辅助 Tensor 和最大合法容量。
- 主动增加复制延迟及不同层执行时间，交替使用内容明显不同的权重发现槽位串层。
- 重复 forward 至少 1000 次，包含 EP1 小逻辑测试和真实 EP8 组合测试；仅使用 DFC 实际支持的规模。
- 前后两次 batch 的 token 数不同；连续 chunk；本地零 token、路由偏斜和 rank 间不同 source rows。
- 当前分支 hc-pre-from-prev/常规路径，以及目标配置实际触发的 IndexCache 和 late-layer tail。
- load 中途失败、请求取消、容量不足、计划失败及正常退出的资源生命周期。
- 既有 FuseEP 与未启用功能的普通模型启动回归。

特殊拓扑和长压力测试放 manual，稳定小规模 NPU 用例按现有注册表选 suite。
server fixture 复用，退出使用 `terminate_and_kill_process_tree`。

### 12.4 性能和内存

对照至少四组：原 `fused_deep_moe`、全 HBM DFC、串行加载 DFC、双缓冲预取 DFC。
后两组使用相同的池、格式、权重、容量和模型配置。
串行模式通过 benchmark 注入策略控制，不增加生产部署开关。

先固定 KV/工作集、chunk 和并发，覆盖短 prompt、小 chunk、目标 8192 chunk、多请求组合；
再固定 HBM 预算评估新增 KV 容量和可承载吞吐。记录：

```text
启动/运行 HBM 峰值，独立专家 backing storage 数量及字节数，DRAM 物理贡献
各层复制字节/时延、ready 等待、submitted 间隙、DFC 和 shared MLP 时延
完整 Prefill 时延、prefill tokens/s、TTFT P50/P99、PD KV 传输时延
与常驻 DFC 相比的计算退化、与串行加载相比的重叠收益
```

硬性验收：无静默精度错误、无槽位覆盖/悬空访问、真实只保留 M0+两个专家槽，
启动也不累计完整专家 HBM；重复请求和 PD KV 传输正确。

性能报告分别给出“能否运行且节省内存”和“目标负载是否值得部署”的结论。
建议首轮观察目标为并发搬运下 DFC 计算时延退化不超过 5%；该数值是评审建议，
不是已经测得的结果，也不代表端到端时延退化一定小于 5%。
若持续带宽不足，报告残余等待和可用工作负载范围，不以更多缓冲掩盖带宽瓶颈。

## 13. 首版决策与当前状态

1. 使用独立 `ascend_dfc` 后端，保留原 FuseEP 作为对照。
2. 首版聚焦 A3 + DSV4 W8A8 + PD Prefill + eager + PP1/MoE-TP1/固定 EP。
3. 首个 MoE 常驻；两槽搬运 routed 和独立 shared 的完整执行包，router 常驻。
4. shared fusion 首版关闭；逐层加载是正式交付的一部分。
5. embedded SHARED 池、每 EP rank 自存分片、一条复制流；不修改外部依赖。
6. 首版不同时使用同一 MemFabric offload 实体承载 HiCache，不支持动态更新和交错 forward。
7. 先兼容预检和全 HBM DFC，再完成卸载、逐层加载和 PD 验收。

2026-10-27 已按当前分支和最新 MemFabric 接口完成首版工作区修改。
已写 CPU 模拟测试、池配置示例和 Prefill 启动脚本；目前不声称 S0–S5 的设备验收通过。
目标 checkpoint、镜像 wheel、A3 精度和内存性能均待验证。
按 DFC 接入和权重卸载分成两个功能提交。第一份独立支持常驻 HBM DFC，
第二份依赖第一份并增加权重池与预取。未编译、安装依赖、执行服务或测试；未推送或切换分支。
