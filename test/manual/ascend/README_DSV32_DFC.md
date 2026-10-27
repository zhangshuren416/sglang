# DeepSeek-V3.2 单机 DFC 验证

脚本：[launch_dsv32_dfc_single_node.sh](launch_dsv32_dfc_single_node.sh)。
在用户已有的 Ascend A3 镜像中运行，镜像需要已安装当前 SGLang 代码和包含
`Buffer.dispatch_ffn_combine` 的 ZBAL。脚本不安装依赖或编译算子。

默认使用一台 A3 的 16 个 NPU rank（通常对应 8 张双 die 卡），TP16 / EP16，
Attention DP1。普通 8 × 64 GB HBM 不足以容纳完整 V3.2 W8A8 常驻权重；
不要把这里的“单机”理解成任意 8 卡服务器。使用本地 ModelSlim W8A8 权重，
路由专家必须是 `W8A8_DYNAMIC` 对应的 INT8 scheme、每通道 scale、全零 weight
offset，且 `config.json` 中 `swiglu_limit` 缺省或为 null。
启动时适配器会进一步检查实际加载的专家配置。

本阶段所有权重常驻 HBM。服务使用普通合设模式；发送 `max_new_tokens=1` 的请求，
可只验证 prefill 前向和首 token。无需 decode 实例、PD router 或 MemFabric SHARED
权重池。这个入口用于先检查 DFC 接入，PD 权重卸载另见
[README_DFC_WEIGHT_OFFLOAD.md](README_DFC_WEIGHT_OFFLOAD.md)。

## 启动

先在已有镜像中完成现有的 CANN 环境初始化，并确认 Python 导入的是包含本次改动的
SGLang。进入本仓库根目录执行：

```bash
# 使用本机已有的 ZBAL 预算（MiB）。58368 是原 ZBAL 混合分配部署示例，
# 应根据实际 HBM 与既有配置调整，不是固定的 DFC workspace 大小。
export SGLANG_ZBAL_LOCAL_MEM_SIZE=58368
export SGLANG_ZBAL_BOOTSTRAP_URL=tcp://127.0.0.1:24669

# 若镜像已指定可见设备，保留原配置；否则在这里指定 16 个可用逻辑 NPU。
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# 只打印命令，不加载模型、不初始化 NPU。
DRY_RUN=1 bash test/manual/ascend/launch_dsv32_dfc_single_node.sh \
    /path/to/DeepSeek-V3.2-W8A8

# 实际启动；终端实时输出，日志同时保存在 ./logs/dsv32-dfc/。
bash test/manual/ascend/launch_dsv32_dfc_single_node.sh \
    /path/to/DeepSeek-V3.2-W8A8
```

DFC 模式默认补上原 ZBAL 部署使用的混合分配配置：
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`、
`ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True`；已有非空值会保留。
bootstrap 的端口要空闲，不与 HTTP 服务或其他作业复用。

默认 `PREFILL_TOKENS=128`，同时设置 chunked prefill 与 max prefill tokens。
对于 V3.2 的 top-k=8、EP16，DFC 扩展行容量是 `128 × 8 × 16 = 16384`。
先通过短请求，再逐步扩大上限；直接套用 8192/16384 等大 token 默认值会显著
增大当前实现预留的工作区。`CONTEXT_LENGTH=2048`，最大并发为 1，KV token
池上限同为 2048；这是一组功能验证配置。

## 发起请求

在另一个终端执行，`/health` 成功后发送请求：

```bash
curl --fail --silent --show-error http://127.0.0.1:30000/health

curl --fail --silent --show-error http://127.0.0.1:30000/generate \
    -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":1},"return_logprob":true,"top_logprobs_num":5}' \
    | tee dfc-short.json
```

再换几条中文、英文和算术提示词重复请求，并发送超过 128 token、低于 2048 token
的提示词，覆盖多次前向和 chunked prefill。首 token 请求通过后，可试
`max_new_tokens=32`，观察合设模式下的多次前向；此时 decode 也经过当前 DFC
路径，只作为功能冒烟，不能据此认为已适配生产 decode 或 PD decode 实例。

## DeepEP 对照

先停止 DFC 服务、确认 NPU 内存释放，再在相同镜像和同一份 checkpoint 上执行：

```bash
MOE_BACKEND=deepep bash test/manual/ascend/launch_dsv32_dfc_single_node.sh \
    /path/to/DeepSeek-V3.2-W8A8
```

该模式使用 `--deepep-mode normal`；需要镜像中已有可用的 Ascend DeepEP。
按现有 DeepEP 部署设置通信环境。脚本只在 DFC 模式补充 ZBAL 分配器配置；
如果此前已在 shell 中手动设置了 ZBAL 相关环境，DeepEP 模式仍会继承，
应与本机原有 DeepEP 对照配置保持一致。
两次运行的 TP/EP、DP、共享专家、量化、输入、采样及 token 上限保持一致。
重复同样的 `/generate` 请求，将结果另存为 `deepep-short.json`，对比生成 token、
`meta_info.output_token_logprobs` 和 `meta_info.output_top_logprobs`。
INT8 实现间可能有数值差异；单条文本相同仅代表冒烟通过，不能替代 MoE 数值检查
或 GSM8K/MMLU 精度回归。

## 常用调整与状态

```bash
# 逐步增加 prefill 预算；仍受上下文、HBM、DFC workspace 与实际调度影响。
PREFILL_TOKENS=256 PORT=30001 \
    bash test/manual/ascend/launch_dsv32_dfc_single_node.sh \
    /path/to/DeepSeek-V3.2-W8A8
```

脚本末尾可追加服务参数，后置同名参数会覆盖默认值。首次验证请保留默认拓扑，
不要追加投机推理、CP、双批次重叠、图执行或权重卸载参数。
如果报 `Installed ZBAL was built without A3 dispatch_ffn_combine`，说明镜像中的
ZBAL 不含当前 DFC 接口，需要换成用户已构建的对应版本。

本脚本只完成静态检查和命令展开检查，尚未在 A3 上启动或执行模型；尚无
DeepSeek-V3.2 + DFC 的设备精度、内存或性能验证结论。
