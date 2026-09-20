# 本地 rollout 优化（2026-09-20）

本次仅修改本地工作区，不推送、不部署到服务器。正在运行的 DD 和 MAPPO-window 实验继续使用 b79334a。

## 修改

- 增加 `BatchRunner.submit_many`：只在启动阶段以临时线程同时启动独立 Session，避免逐个等待进程引导和大 RunSpec 传递。并发仿真进程数仍受 workers 限制；运行时仍用管道 readiness 收集响应，没有新增持久的接收线程。启动失败隔离、重复 ID 原子校验、部分成功的资源清理都有测试。
- 环境重置拆成准备工作负载、批量启动、完成 reset 三步。保留相同 seed/slot/generation、冷缓存与显式历史清空；仍复用未推进的 spec probe，不复用已经推进的 episode。
- 批量采样直接将固定 agent 的观测、奖励、done/terminated/truncated 和全局 state 组装为 TensorDict，省去每个 worker 每一步经过通用 PettingZooWrapper 的重复转换、动作 mask 更新和数据复制。PettingZoo 环境及 wrapper 的 spec/独立评估接口保持可用。
- 整批动作一次从 Tensor 提取为 NumPy，再分配到各 worker。

没有改变请求生成、物理机制、奖励、学习算法、动作、训练预算或周期同步屏障。没有实现常驻 SimGrid 引擎的跨 episode 重置；新 episode 仍启动新进程，只是并行启动。

## 测量方法

从本地 HEAD b79334a 提取原版学习包和 SDK 作为独立参考路径，和当前实现交替运行。两种动作接口各重复三次，8 workers，small-window 场景，seed 0，256 个向量步，即 2,048 个环境步。每次采样含一次完整 episode 批量重置；setup 单独计时。固定 Torch seed 的随机动作，不执行策略网络推理或训练更新。正式对比关闭 cProfile，以免 Python 调用次数变化造成剖析器开销偏差。

原始文件与复现脚本位于 `outputs/rollout-local/`。本机与 Linux 服务器硬件不同，吞吐的绝对值不可直接比较；最终服务器增益需要后续部署后实测。

## 正确性

新增测试验证快速打包结果与原 PettingZooWrapper 的观测、全局 state、奖励、done/terminated/truncated 完全一致，并覆盖 DD、历史阈值策略和只重置部分 worker 的情况。原有 CPU 训练、恢复、GRU 梯度、随机/确定性串并行评估测试同时执行。

## 本机结果（无 profiler，三次均值）

| 动作接口 | 优化前步/秒 | 优化后步/秒 | 吞吐提升 | 批量重置前→后 |
|---|---:|---:|---:|---:|
| MAPPO-no-context | 309.0 | 370.7 | 20.0% | 1.713 → 1.011 秒 |
| DD-adapted | 291.9 | 361.8 | 23.9% | 1.772 → 0.989 秒 |

每种动作接口的六次新旧运行处理的终结请求总数一致。完整均值、范围和启动耗时见 `outputs/rollout-local/comparison.json`。

全仓验证：`python -m pytest -q` 为 **194 passed、1 skipped**（无 NVIDIA GPU 的 CUDA 测试）；Ruff 和 `git diff --check` 通过。CPU 训练与断点恢复包括在测试内。本次保留为本地未提交改动，没有推送或部署。
