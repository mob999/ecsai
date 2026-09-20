# DD-adapted 固定维度请求决策基线

对应原论文 IV-A 的 Dimension-Determined (DD)：预设最大请求数，为可见请求分别输出决策。旧 ZIP 没有完整 DD 实现，论文没有给出其完整网络、训练器及周期内到达处理细节。因此本实现是 **DD-adapted**，不是参考文献 [9] 的逐项复现，也不是去掉 GRU 的 DEPPO。

## 观测与动作

每个 agent 固定 `scheduler_capacity + 1` 个请求槽位（FIFO 等待项加一个活动调度项）；默认 101。槽位按活动项在先、等待 FIFO 在后排列，在每次观测时重新绑定请求 ID，控制 DTO 传请求 ID 而不传槽位索引。输入只有当前可见请求，不读取未来到达时间表。

当前 13 维局部统计观测之后，每个槽位包含归一化大小、归一化剩余期限、已转发标记、有效位。空槽位零填充。默认每个 actor 输入 417 维，输出 102 维动作：101 个请求决策分数加一个回源带宽比例。请求分数范围 [-1,1]，非负表示转发；比例范围 [0.05,0.95]。本版本沿用 BenchMARL 的有界连续分布，以分数的符号执行二元决策，**不是 Bernoulli 离散策略**；这是明确的适配差异。

输出不再是共享决策函数的 ω、b。网络一次性输出各槽位的直接决策，仍使用独立 actor、共享中央 critic 和 MAPPO。actor 为两层可配置宽度的 ReLU MLP，无 GRU；critic 继续使用所有 agent 的 13 维统计观测。DD 额外获得当前逐请求信息，不能将对比解读成严格等信息消融。

空槽位以及已转发槽位使用固定分布，输出不依赖网络参数，因此没有策略梯度、熵梯度或 KL 贡献。其固定密度在 PPO 新旧概率比中抵消。已转发请求仍受原有一次转发限制，并本地处理。动作和采样 log probability 继续由现有 PPO 稳定性检查验证。

## 时间、排队和公平性

新增 SDK 领域配置 `scheduler_release="window"`。窗口开始时只有此前已到达的请求获得本周期调度资格；周期内新到达请求进入同一个有限 FIFO，等待下一窗口。它们仍从原始到达时计时，受原有队列容量、溢出与截止规则约束，没有额外无限缓冲区。转发请求无需第二次策略决策；仍按目标 FIFO 排队。

默认连续模式的行为不变。`train --method DD-adapted` 自动将 release 设置为 window，并将最终配置写入 checkpoint/config。环境直接构造时需显式设置 window。槽位数覆盖所有可能的调度等待/活动项，不额外丢弃“超出神经网络输入”的请求。

周期边界上的新到达请求仍遵循 SDK 原有规则：当前快照不包含它们；连续策略可用本周期参数处理，而 DD 要到下个可见快照才能决策。这是批处理机制的代价，应单独报告，不能全部归因于策略表示。

因此应同时比较：

- DD-adapted 与 `MAPPO-no-context`，两者都使用 `small-window.json`：同批量时序的主要对照。
- 原有 continuous 阈值策略：实际端到端性能对照，包含实时规则执行的优势。
- 有/无 GRU：独立的历史编码消融。

请求轨迹、网络、缓存、队列容量、截止时间、WRR 本地目标、最低广播负载转发目标、带宽预算和奖励不变。训练预算按环境决策步计算。记录训练/推理耗时、成功率、延迟、溢出，以及评估的 `policy_joint_calls`、`policy_agent_calls`；调用计数含排空阶段。

评估停止到达后继续排空。DD 以及 window 模式的学习策略在排空时继续推理处理新可见请求，但不追加训练步或 episode reward。原有 continuous 策略继续沿用原来的固定最后控制排空语义。不同策略的 checkpoint 不混用，恢复检查方法、场景、动作维度和训练选项。

## 本机及服务器入口

```bash
# 默认槽位数的 CPU smoke；生成本地 CSV、checkpoint、W&B offline 文件。
uv run --package edge-sim-learning edge-learn train \
  --profile smoke --method DD-adapted --episodes 2 --workers 1 \
  --batch 8 --minibatch 8 --epochs 1 --eval-interval 8 --eval-episodes 1 \
  --device cpu --wandb-mode offline --output outputs/dd-smoke

# 正式对照使用同一个场景、网络宽度和训练预算。
uv run --package edge-sim-learning edge-learn train \
  --scenario configs/learning/small-window.json --method DD-adapted \
  --episodes 2048 --workers 4 --hidden-size 256 --device cuda \
  --eval-workers 4 --eval-stochastic --wandb-mode online --output outputs/dd-seed0

# 将上一条 method 改为 MAPPO-no-context，output 改成独立目录，得到匹配批量时序对照。
# seed 0/1/2 分别运行，不将新批量时序结果与旧 continuous 实验混为同一场景。

uv run --package edge-sim-learning edge-learn evaluate \
  --checkpoint outputs/dd-smoke/best.pt --episodes 2 --workers 2 \
  --wandb-mode offline --output outputs/dd-smoke-eval
```

`--resume outputs/dd-smoke/last.pt` 配合更大的 `--episodes` 总预算可继续训练。调参使用 validation，最终 test 保持隔离；本次实现验证不代表 DD 已收敛或优于任何方法。

## 验证

`tests/test_dd.py` 覆盖槽位可见性、请求 ID 绑定、重置、排空、环境合同、截断、屏蔽槽位梯度、CPU 更新、保存恢复、确定性/随机策略串并行评估一致性，以及相同批量时序下 DD 与固定阈值动作的物理结果一致性。原有 continuous 模式回归测试同时保留。

本机验证（2026-09-20）：`python -m pytest -q` 为 189 passed、1 skipped（无 NVIDIA GPU 的 CUDA 测试）；Ruff 检查通过。默认 101 槽位的 CLI smoke 完成 16 个环境步、两次更新批次，并通过双进程 checkpoint 评估；结果位于本地 `outputs/dd-smoke/` 与 `outputs/dd-smoke-eval/`，只用于闭环验证，不作为收敛结论。
