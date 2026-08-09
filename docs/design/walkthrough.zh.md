# 完整形式化推导示例

这个示例跟随一个 row-parallel Transformer MLP 片段从 architecture-independent work 到 hardware-bound simulation plan 的过程。它足够小，可以逐项检查，同时覆盖 semantic work、tensor parallelism、collective communication、portable mapping、architecture binding、scheduling、simulation、optional emission 和 profiler lineage。

!!! note "实现状态"
    到 `PortablePlanIR` 的过程对应当前 Transformer slice。后续示例是目标设计的说明性伪代码：当前 `ConcretePlanIR` 只有 experimental common schema，typed target extension、scheduling producer、timing projection 和 MachineIR producer 均未实现。

## Source Problem

考虑 row-parallel linear operation，symbolic token count 为 (M)，输入宽度为 (N)，输出宽度为 (K)，tensor-parallel degree 为 (P)：

```text
Y[M, K] = X[M, N] × W[N, K]
```

每个 rank 拥有 (N/P) 个输入列，计算 partial output，并参加 AllReduce。精确 local matrix work 是 `2 × M × (N/P) × K`；logical collective payload 是 `M × K × element_bytes`。

## Stage 1：ModelIR

Frontend 在不知道 distribution 和 hardware 的情况下建立模型含义：

```text
%y = model.linear %x, %w
    : tensor<MxNxbf16>, tensor<NxKxbf16>
      -> tensor<MxKxbf16>
```

这一阶段之后已知：

- value 和 operation identity；
- shape、dtype、state role 和 effect；
- `x`、`w` 与 `y` 之间的 semantic relation。

仍然未知：TP ownership、communication、recomputation、target kernel、placement、queue、buffer 和 time。

Verifier 检查 definition/use、type、symbol、effect 和 stable ID。Profiler checkpoint 可以比较 shape 或 numerical reference，但不能附加 target duration。

## Stage 2：DistributedTaskIR

Distribution pass 绑定 TP strategy，并引入 logical rank 和 communication：

```text
mesh @tp<P>

task @local_linear rank=%r {
  %partial = compute.matmul %x_shard[%r], %w_shard[%r]
  %y = collective.all_reduce %partial group=@tp reduction=add
}
```

Pass 决定 logical ownership 和 dependency。它不选择 NCCL、route、physical device 或 collective algorithm。

Verifier 证明 shard 可以重建 source tensor、每个 rank 属于 mesh、collective participant 匹配、result 保持原 shape，并且 operation/message volume 守恒。

## Stage 3：PortablePlanIR

Portable planning 显式表达 execution intent，同时保持 target independence：

```text
portable.task @gemm {
  depends_on = []
  work = {operations = 2*M*(N/P)*K, reads = ..., writes = ...}
  requires = [matrix_engine(dtype=bf16)]
  concurrency_group = "compute"
}

portable.task @reduce {
  depends_on = [@gemm]
  work = {message_bytes = M*K*2}
  requires = [collective(kind=all_reduce, reduction=add)]
  concurrency_group = "network"
}
```

Abstract boundary buffer 携带 size、alignment、role 和 lifetime constraint。Implementation requirement 描述 capability，不指定 kernel 或 vendor library。

在当前 production boundary，这份 snapshot 已经完整并经过验证。`estimate_block()` 可以从它派生 cost，而不向 IR 写入 duration。

## Stage 4：Target Binding

Target gate 消费 `TargetProfile`、`DeploymentProfile` 和 evidence policy。对于 GPU deployment，它可以选择 GEMM implementation 和 ring AllReduce；对于 LPU，它可以选择不同的 matrix primitive 与 collective protocol。

```text
@gemm   -> implementation gpu.cublas.gemm.bf16.v17
@reduce -> implementation nccl.all_reduce.ring.v3
abstract compute -> device gpu:0, engine compute:0
abstract network -> communicator tp0, channel 2
```

Gate 检查 capability、dtype/layout limit、library/ABI revision、topology 和 capacity。缺失 evidence 不会静默变成零；resolver 会记录 interpolation、extrapolation 或 analytical fallback。

## Stage 5：ConcretePlanIR

Scheduling 与 memory planning 固化权威 execution-plan envelope。下例展示 queue-centric GPU common core；若 target 是 spatial/dataflow LPU，还必须有 typed target extension 表达 route、issue 或其他 target-only constraint：

```text
buffer %partial {device=gpu:0, space=hbm, offset=0x4000, size=M*K*2}

command @c0 Launch {
  implementation = gpu.cublas.gemm.bf16.v17
  queue = gpu:0/compute:0
  writes = [%partial]
}

command @c1 Collective {
  implementation = nccl.all_reduce.ring.v3
  queue = gpu:0/network:0
  waits_for = [@c0.completed]
  reads_writes = [%partial]
}
```

Common correctness 由 dependency、ordering、token、buffer 和 resource constraint 定义，不需要 predicted start time。Target-enforced cycle/slot 则属于 typed extension，而不是 prediction。

完整 producer 的 verifier 必须检查 DAG acyclicity、ordering legality、synchronization、buffer lifetime/reuse、placement、ABI compatibility、capacity 与 target extension。当前 structural verifier 尚未覆盖这里列出的全部义务。

## Stage 6：Simulation 与 MachineIR

Simulation 分支解析 duration interval，并执行 concrete readiness rule：

```text
@c0 predicted [0.000 ms, 0.310 ms]
@c1 predicted [0.310 ms, 0.442 ms]
critical_path = [@c0, @c1]
```

Program 分支把同一批 command lower 为 target instruction 或 runtime call：

```text
machine.call @gemm_executable(...)
machine.signal @c0.completed
machine.wait @c0.completed
machine.call @nccl_all_reduce(...)
```

时间区间是可重建 annotation。Machine instruction 和 simulation event 都引用 `@c0`、`@c1` 以及它们的上游 lineage。

Concrete digest、time interval、simulation event、evidence/policy fingerprint 与 diagnostic 一起组成 `TimelineBundle`。Bundle 只引用权威 command plan，不能产生自己的 schedule。

## Stage 7：Observation 与 Recalibration

Runtime profiling 记录实际 completion 和 counter：

```text
Observation(command=@c0, duration=0.337 ms, source=runtime-profiler-r8)
Observation(command=@c1, duration=0.151 ms, source=runtime-profiler-r8)
```

Normalization 产生 immutable `ObservationSet`。Calibration 可以发布新的 target-scoped evidence revision，planner 随后重新 estimate 并生成新 plan。原有 IR、cost view、simulation report 和 program artifact 仍然可以复现。

## 这个示例建立了什么设计约束

这个说明性示例要求每个阶段消除一种 unknown，只引入自己拥有的信息，并保持 stable lineage。Exact work 不会成为 fitted parameter；target cost 不会成为 model semantic；predicted time 不会成为 execution dependency；runtime observation 不会修改旧 plan。只有 production producer、verifier 和 cross-consumer test 到位后，这些约束才构成实现证据。
