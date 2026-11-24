# 作业三：分层缓存与 Host Offloading（进行中）

## 成员贡献

**Steven Walxim（2501112130）**负责本分支的作业三需求分析、实验设计、
off/on 对照运行器、逐请求 token 与来源一致性复核器、独立后台实验启动器、
本地回归测试、完成事件/调度状态观测器、本人账户下的 Ascend 实验和证据归档。已完成 0.8B TP1 的
baseline、强制 decode 恢复及强制 prefill 恢复 smoke；两组对照共核对
16 个请求、256 个输出 token，全部一致。新增脚本与实验记录属于 Steven
本阶段的贡献，具体代码和证据可由本分支提交记录复核。

**复用边界：**TokenSpeed 上游的 Host L2、快照池、传输与调度机制，以及小组
HW1/HW2 已有的 Ascend/模型适配继续沿用；不将这些已有机制计为 Steven
从零实现。其他成员的 HW3 实际贡献须由本人/小组补充，本稿不推定。

## 当前完成状态

本稿是进度报告，**尚未达到作业三全部完成要求，也尚未提交教学网**。
代码已在本地 feature branch 分阶段提交；本轮 HW3 提交尚未推送 GitHub。

| 要求 | 当前证据 | 状态 |
| --- | --- | --- |
| 同一请求暂停/恢复正确 | 0.8B TP1，decode 与 prefill 各 8 请求，逐 token 对照通过 | 初步通过；新增 172 次 snapshot/restore frontier 一致性检查 |
| 多会话隔离 | 4 个不同前缀，交替运行、重复访问、扩展前缀，逐请求输出一致 | 初步通过 |
| 实际卸载、HBM 块回收及复用 | 完成 ACK、双向 payload 字节数、最终 16/16 device blocks 可回收；各模式 arena 204 MiB | TP1 已观测；私有块复用因果证明仍待补 |
| Host 容量、淘汰、未命中 | 显式 L2/snapshot 预算配置；尚未校准淘汰压力 | 待验证 |
| 重复恢复无持续泄漏 | 2 周期后无活跃请求、无 pending transfer、snapshot 全部释放 | 静止状态检查通过；长周期待验证 |
| 0.8B 与更大模型，覆盖 TP2 | 已完成 0.8B TP1；目前只有单 chip 空闲 | TP2 与较大模型待验证 |
| 高/低复用、两种 Host 预算、吞吐/恢复代价 | 18 次独立 Engine 对照及 12 组 token/trace 复核通过，force interval 0 | 已完成所测配置；未观察 Host 复用收益 |

## 已确认的首次硬件实验

配置为 0.8B BF16 权重/KV、FP32 递归状态、TP1、物理 chip 1，
固定 16,384-token 配额、4 会话、2 周期，输入长度 256/264、输出长度 16。
三种模式的实际 device arena 均为 213,909,504 bytes（204 MiB），
16 个可用 LCM blocks。off 禁用两个 Host 池；on 的 L2/snapshot ratio 均为 1。

强制 decode/prefill 的输出分别与 off 的 8 个请求完全一致。日志中的
179 次/4 次 snapshot store、restore **开始**记录不能代替传输完成统计；
当前结果明确保留 `offload_lifecycle_verified: false`。
强制恢复会改变调度行为，这些运行仅用于正确性，不用于声称性能提升。

最初的 1,024-token 配额只提供两个 hybrid-cache pages，无法接纳请求；
保留失败证据后，将三种模式统一调整为已有 HW2 验证过的 16,384 配额。
这不是“关闭 offloading 容量更小”的实验结论。

49 个下载文件逐一校验 SHA256；两个 token 对照在本地再次独立复核。
全部任务完成后确认 chip 1 无本轮进程。详见
[硬件结果](results/2026-11-24-tp1-smoke/summary.json)、
[缓存生命周期与混合状态设计](docs/implementation-plan.md) 和
[运行说明](README.md)。后续实测应增补此报告，不能以计划替代结果。

## 已确认的完成事件与静止状态实验

独立 instrumented smoke（source `154b5a6`）三种模式均通过原始 token
对照和 trace 检查；输出也与首次未加观测器的 smoke 一致。另行下载的 51 个
证据文件逐一核对 SHA256，完成事件及 token 对照均在本地再次复算。

| 模式 | 完成 D2H payload bytes | 完成 H2D payload bytes | 完成 snapshot/restore 各次数 | 恢复 batch 观测中位数 |
| --- | ---: | ---: | ---: | ---: |
| off | 0 | 0 | 0 | 不适用 |
| decode 强制恢复 | 2,659,318,992 | 3,483,303,120 | 165 | 7.817 ms |
| prefill 强制恢复 | 583,238,640 | 273,531,888 | 7 | 68.734 ms |

时间从 Host 方法入口到观测到 ACK，包含排队、轮询与记录开销，**不是纯 DMA
设备时间**；重叠传输的耗时不能直接累加为 wall time。字节数为实际 rank-local
字段 payload 与 slot blob，不计 allocator padding。两个 on 模式各分配
201,326,592 bytes 逻辑 L2 tensor、201,326,592 bytes 逻辑 snapshot tensor，
另有 432-byte slot blob；不是物理 pinned allocator reserved memory 或 RSS。

所有 172 次恢复的记录 frontier 与 snapshot 一致。最终每种模式均 16/16
设备块可回收、0 活跃请求、0 pinned Host entries、0 pending transfers；on
模式 snapshot 16/16 blocks 空闲。Host L2 仍保留可淘汰缓存是预期行为。
这支持本次有限周期无持有资源残留，不能推广成任意长周期无泄漏。

观察到的 Host 物理位置换键（decode 358、prefill 35）不是完整逻辑淘汰
计数；设备 block 的请求 owner 变化也可能来自前缀共享，不能单独证明私有
块被回收再使用。前缀命中包含所有缓存层，不能当作 Host 专属命中率。
本轮为强制恢复正确性实验，不声称吞吐提升；完整作业标志仍为 false。
详见 [完成事件证据](results/2026-11-24-tp1-trace/summary.json) 与
[观测口径](docs/runtime-observer.md)。

## 补充调度器契约测试

服务器现有 native scheduler bindings 上执行上游
`test_retraction_image.py` 与 `test_kv_cache.py`，共 **63 passed**。
测试文件及 conftest 与本地源码逐一 SHA256 一致；涉及 L2 前缀与 slot blob
分腿完成、ACK 前禁止恢复、结束请求释放 pin，以及混合 cache groups 的
快照恢复。这些是复用的上游测试，由 Steven 执行及归档，不计为新增实现。
测试以模拟执行事件/ACK 驱动真实 native scheduler，不是 NPU DMA 或模型
数值测试，不能代替 TP2/大模型验证。
[记录与测试源码哈希](results/2026-11-24-scheduler-contracts/summary.json)。

## 正常调度下的容量与性能对照

0.8B TP1，8 会话、2 周期，高/低复用各执行 off、L2 ratio 0.5、L2 ratio 1，
每种配置 3 次独立 Engine，共 18 次运行。force interval 固定 0；on 的
snapshot ratio 固定 1。193 个下载证据文件逐一 SHA256 校验，18 份原始 trace
在本地重新计算，12 组输出对照再次通过：192 个请求、3,072 个输出 token。
所有模式实际 device arena 都为 204 MiB、16 可用 LCM blocks。

Host 小预算为 96 MiB 逻辑 L2 + 192 MiB snapshot，大预算为 192 MiB L2 +
192 MiB snapshot，on 均另有 432-byte slot blob。snapshot 池在正常调度本轮
没有使用；物理 pinned allocator reserved memory / RSS 不在这些数字内。

每个运行的 16 个请求都首次接纳未命中：**0 prefix hits、0 completed H2D
loadbacks**。on 产生 writeback 和 Host 物理位置换键，但所测“高复用”访问
模式在这些预算下仍未成功命中。off 在相同设备预算下通过重算完成，不能
声称 off 容量不足，也不能声称本轮观察到 offloading 加速。

下表仅列 264-token 后续访问阶段；低复用阶段是同长度的新会话。首次冷启动
与其他 256-token 初访分别记录在 analysis.json，未混合计时。

| 访问模式 | Host L2 ratio | 三次运行输出吞吐中位数（tokens/s） | 对 off 的配对吞吐比中位数 |
| --- | ---: | ---: | ---: |
| 高复用 | off | 9.617 | 1.000 |
| 高复用 | 0.5 | 9.347 | 0.972 |
| 高复用 | 1 | 9.619 | 1.000 |
| 低复用 | off | 9.967 | 1.000 |
| 低复用 | 0.5 | 9.911 | 0.994 |
| 低复用 | 1 | 9.821 | 1.002 |

每个 phase 的吞吐 = 该 phase 输出 token 总数 / batch 耗时总和。
配对比先对每次独立运行计算，再取中位数，因此不等于表中两列中位数之比。
三次运行执行顺序固定、共享服务器负载可能漂移，且含观测器开销；这些小
幅变化不能解释为统计确定的改善或退化。要展示适用条件，下一步需选更大
Host 预算或调整工作集，并先确认实际 Host 命中，再评价传输与重算的权衡。
[完整分析](results/2026-11-24-tp1-matrix/analysis.json)；
[校验摘要](results/2026-11-24-tp1-matrix/summary.json)。
