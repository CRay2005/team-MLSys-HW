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
| 高/低复用、两种 Host 预算、吞吐/恢复代价 | 18 次独立 Engine 对照正在运行，force interval 0 | 结果待复核 |

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
