# 作业三：分层缓存与 Host Offloading（进行中）

## 成员贡献

**Steven Walxim（2501112130）**负责本分支的作业三需求分析、实验设计、
off/on 对照运行器、逐请求 token 与来源一致性复核器、独立后台实验启动器、
本地回归测试、本人账户下的 Ascend 实验和证据归档。已完成 0.8B TP1 的
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
| 同一请求暂停/恢复正确 | 0.8B TP1，decode 与 prefill 各 8 请求，逐 token 对照通过 | 初步通过；完整状态边界证据待补 |
| 多会话隔离 | 4 个不同前缀，交替运行、重复访问、扩展前缀，逐请求输出一致 | 初步通过 |
| 实际卸载、HBM 块回收及复用 | 日志记录 writeback、snapshot、restore；设备 arena 各模式均 204 MiB | 完成事件、块级复用与字节统计待补 |
| Host 容量、淘汰、未命中 | 显式 L2/snapshot 预算配置；尚未校准淘汰压力 | 待验证 |
| 重复恢复无持续泄漏 | 2 个访问周期的 smoke | 长周期与静止状态检查待补 |
| 0.8B 与更大模型，覆盖 TP2 | 已完成 0.8B TP1；目前只有单 chip 空闲 | TP2 与较大模型待验证 |
| 高/低复用、两种 Host 预算、吞吐/恢复代价 | 已有可复现实验运行器 | 正式对照待运行 |

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
