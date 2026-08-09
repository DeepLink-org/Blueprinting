# Vidur 原始组件 Profile 对齐

这个实验回答一个刻意可证伪的问题：**Blueprinting 独立完成推理 phase 的 lowering 和 cost evaluation 之后，其 component cost 与兼容的 Vidur profile 相差多少？** Vidur 是 reference result，不是 production analysis path 的实现依赖或 latency provider。

## 实验边界

执行顺序固定为：

```text
Transformer semantics + mapping + phase context
  -> Blueprinting ModelIR
  -> Blueprinting DistributedTaskIR
  -> Blueprinting PortablePlanIR
  -> Blueprinting peak-only and system-evidence costs
  -> freeze plan digests and compiled estimates
  -> exact Vidur baseline lookup
  -> coverage and error report
```

`InferenceCostProvider.resolve()` 是 Blueprinting 自有性能数据库或硬件仿真器的扩展点；`InferenceBaseline.lookup()` 是外部 oracle 接口。`VidurProfileBaseline` 只实现 `lookup()`，因此不能被意外传入 `estimate_inference_phase()`。

独立的 `VidurProfileImporter` 可以显式把用户提供的 profile row 转换成 `PerformanceDatabase`。这是另一条 workflow，也是一项明确 policy decision：只有用户刻意把该 database provider 安装进 `CostResolver`，它才会影响 costing。本实验仍只使用 `VidurProfileBaseline`，因此 oracle isolation 不变。

Experiment report 会记录 `oracle_read_during_lowering = false`、`oracle_read_during_costing = false` 与 `fit_against_case_outputs = false`。Per-case correction factor 和 Vidur duration 都是 lowering/costing 的禁止输入。

## 对比契约

Adapter 对 model/hardware identity、dtype、model dimension、maximum sequence length、TP degree、batch/token shape、phase、context、attention backend 与 cache block size 做 exact lookup，并记录输入 CSV 与固定 upstream revision 的组合 digest。

Vidur 的 decode `kv_cache_size` 是当前 token 加入前已经缓存的 token 数；Blueprinting 的 decode `context_tokens` 是加入当前 token 后 attention 可见的 key 数。因此精确关系是：

```text
vidur.kv_cache_size = blueprinting.context_tokens - 1
```

不允许 nearest-neighbor match、interpolation 或 silent zero fill。缺失或语义不兼容的 component 保持 `not-covered`。

## 报告指标

每个 phase report 暴露：

- component 总数、matched 数量与 semantic coverage；
- Blueprinting 完整 block cost；
- Blueprinting 在 matched component 上的 subtotal；
- Vidur 在相同 component intersection 上的 subtotal；
- 未进入比较的 compiled cost；
- component 与 comparable-subtotal 层的 signed absolute/relative error；
- 不可相互抵消的 component MAPE 与最大 component error；
- model、distributed plan、portable plan、hardware evidence 与 baseline revision。

实验同时比较 `PEAK_ONLY` 和 `SYSTEM_EVIDENCE` estimate，用于区分 workload lowering 错误与 target-wide efficiency curve 错误。

## 回归漂移门禁

仓库保存了来自 Vidur commit `8383d2935bc62723a212090baa9f98ada206fc14` 的最小离线 validation slice：A100 上的 Phi-2、TP1，覆盖 128-token prefill，以及 visible context 为 33 和 129 的 decode。Fixture manifest 记录 upstream path、Git blob ID、MIT license、行列投影规则和每个本地 CSV 的 SHA-256；完整 Vidur corpus 仍保持外部依赖。

`data/validation/baseline_regression_contract.json` 固定两类约束：

- semantic intersection 与 comparable-subtotal drift budget：coverage 至少 75%，comparable-subtotal MAPE 不超过 24%，单 case 最大 comparable-subtotal error 不超过 28%，并且相比 peak-only 至少改善 10 个百分点；
- reviewed golden：model/distributed/portable digest、baseline revision、comparable subtotal、逐 case error、aggregate comparable-subtotal MAPE、component MAPE 与最大 component error。

当前三个 case 的 system-evidence comparable-subtotal MAPE 为 23.58%，但不可相互抵消的 component MAPE 是 54.08%，最大 component error 是 99.50%。较低的 subtotal 数值包含显著的跨 component 误差抵消，不能被描述为 component accuracy，更不是 Vidur end-to-end accuracy。这个 gate 只声明 provenance、semantic coverage 与 drift detection。合法改进必须更新 reviewed golden，并收紧有因果意义的 component error；不允许自动执行“接受当前输出”。

本地执行与 CI 相同的强制 gate：

```bash
uv run pytest -m baseline_regression tests/regression
```

## 已知语义缺口

当前 inference dialect 建模一个 dense-MHA、non-gated-MLP decoder template。对于 GQA/MQA 或 gated-MLP model 的 Vidur profile，不兼容 component 会被明确拒绝。Adapter 尚未消费 Vidur all-reduce、send/receive 与 CPU-overhead profile。

固定的 Phi-2 数据是 raw component profile，并不能证明 Blueprinting 已复现 Phi-2 的完整 decoder topology。当前 model schema 还没有表达 norm placement、parallel-residual structure、fusion/layout choice 与 embedding/LM-head work。因此 experiment policy 显式记录 `topology_equivalence = not-claimed-by-raw-component-profile-alignment`。

Vidur 公开的 block aggregation 只贡献一个 `add_time`，而 Blueprinting 将 attention residual 与 MLP residual 保留为两个显式 operation。只有最终 MLP residual 拥有直接的 Vidur component peer；另一个仍作为 excluded compiled work 可见。系统把它报告为 semantic coverage gap，而不会通过重复使用同一个 baseline value 来掩盖差异。

## 如何改进对齐

误差应在最低的因果层修复：

1. workload audit 错误时，修正 operation、byte、KV 或 collective 推导；
2. FlashAttention、paged attention、fusion 或 kernel family 改变 cost 时，引入显式 implementation choice；
3. 使用版本化 observation 与 held-out validation 改进 target-wide size/shape response model；
4. runtime、topology 或 concurrency 影响结果时，扩展 normalized context；
5. 两套系统没有描述同一个 semantic object 时，保持 uncovered。

下一步 coverage 是 gated MLP 与 GQA/MQA lowering，随后接入 collective profile。Scheduler 和 queueing 的对齐属于之后的 serving discrete-event experiment，不属于这个静态 phase baseline。
