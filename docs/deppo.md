# DEPPO v2：共享带宽与稳定训练

本实现复现论文的统计观测、历史 GRU、参数化转发和带宽分配机制。合成请求与修正后的物理模型不用于声称复现论文表格数值。旧四维实验仍保留在历史输出目录；v2 必须从头训练，不兼容四维 checkpoint。

## 场景与 SDK

每个缓存节点总容量 `B_j` 固定，所属调度 agent 每 0.1 秒输出 `r`，回源容量 `r B_j`、交付容量 `(1-r) B_j`。这是 TDD 的流体容量抽象，无无线干扰/衰落/包级协议。不同缓存节点独立，不添加共享骨干。SimGrid 管理传输进度，周期边界更改原生链路容量，不重建活动传输。累计容量按时间分段积分。

SDK `ContentServiceSpec.bandwidth_mode` 默认 `independent`，保持已有用户的独立链路行为；`shared` 要求每个 `CacheNodeSpec.total_bandwidth_bytes_s` 有值且链路独占。`SchedulerControl.backhaul_ratio` 默认 0.5，范围 `[.05,.95]`。`coalesce_backhaul` 默认 true 保持 SDK 兼容；学习场景设 false，实现每请求独立回源。共享内容的存储预留仍只计一份。

学习默认每类传输池 1 活动项、50 等待项，FIFO；调度等待 100，服务时间 1ms。保留 LRU 冷缓存、100MB 容量、完整对象下载、原始截止时间 1s、一次转发。配置并发数可恢复并发池。

三档规模为 `(3,10,300)`、`(5,20,1000)`、`(7,30,2000)`，分别表示集群、缓存和请求/s。内容目录 1000，Zipf 1.0，大小 0.5–7.5MB。拓扑种子固定 1729，目录/请求使用独立 RNG 流。缓存带宽权重 Uniform(1,3)，按 `request_rate * mean_size / delivery_load` 归一化总容量，默认理论交付负载比 .75，可选 .55/.95。每次运行另外报告实际目录热度加权的交付负载和冷缓存回源负载；前者不包含回源，不能理解为完整资源负载。

`configs/learning/paper-audit.json` 按原文 12–36 Mbps 换算为 bytes/s，可能严重过载，仅用于容量审计；主实验使用校准配置，不暗改到达率。

## 观测、动作和奖励

当前观测 13 维：调度队列占用率、平均回源/交付负载、大小/剩余期限各 5 档频率直方图。actor 仅看本地；critic 拼接所有 agent 当前观测。观测打包为 158 维：13 + 8×(13+5) + 1 有效长度。

每个 actor 输出 `(w1,w2,w3,b,r)`，前四维 `[-5,5]`，比例 `[.05,.95]`。请求转发由 `w·x+b >= 0` 决定，等价 sigmoid≥0.5。目标仍为最近广播的最低负载集群，本地按固定总容量加权轮转。GRU(18,64) 编码最近 8 个完成的观测—动作对，拼接当前观测，再经 ReLU MLP(128,128)。独立 actor、共享中心 critic。MAPPO-no-context 只去掉历史编码。

业务奖励：

`(completed - timed_out - rejected - .1 * sum(success_latency/deadline)) / max(1, request_rate*period)`

请求仅在结束时结算一次，无请求结束为零。同步记录 `paper_reward=.5*window_success_rate+.5*actual_utilization`；`reward_mode=paper` 可做消融。利用率是实际传输字节除以容量时间积分。截断请求不伪装超时；训练正确 bootstrap。评估停止产生请求后排空，成功率含排空结果；episode return 只含前 128 个决策步，不包含排空奖励。

## 训练与恢复

默认 lr 1e-4、gamma .99、GAE .95、clip .1、entropy .001、梯度范数 .5。4 个 SDK worker，每批 512 环境步，整批更新 5 轮；不额外套 BenchMARL 多进程向量化。actor 潜在均值限制在 ±3，标准差经有限范围映射；保持 TanhNormal 的有界采样和 log probability 一致。

每个 actor 计算旧/新高斯分布的解析 KL（两者使用相同 tanh/仿射变换）；任一 actor KL 超过 .02，停止该批剩余策略更新，critic 继续。记录每维动作均值/标准差/饱和率、loc/scale、各 actor KL、策略更新次数、梯度和损失。首次更新前检查行为策略 log probability 重算一致。

按用户要求采用简单恢复：初始保存和每个批次完成后原子保存 `last.pt`（含优化器/RNG）；最佳验证模型保存 `best.pt`。出现非有限损失、梯度或参数时停止，写 `failure.json`，不覆盖健康 checkpoint。没有逐更新模型副本或自动回滚。用同场景/种子/worker 数续训，最多损失当前一个批次。

```bash
uv sync --all-packages --locked
.venv/bin/python -m edge_sim_learning.cli train \
  --profile small --output outputs/deppo-v2 --device cuda --wandb-mode offline
.venv/bin/python -m edge_sim_learning.cli train \
  --profile small --output outputs/deppo-v2 --device cuda --wandb-mode offline \
  --resume outputs/deppo-v2/last.pt
.venv/bin/python -m edge_sim_learning.cli evaluate \
  --checkpoint outputs/deppo-v2/best.pt --split test --episodes 30 \
  --output outputs/deppo-v2-test --wandb-mode offline
```

W&B 项目 `ecsai-deppo`，entity 使用 `WANDB_ENTITY`，明确选择 offline/online。CSV 同步保存，环境步是主横轴，agent 步另记。

## 公平实验入口

```bash
# 本地完整流程的小预算验收（不证明收敛）
.venv/bin/python scripts/run_deppo_v2.py --smoke --output outputs/deppo-v2-smoke
# Linux/NVIDIA 正式实验
.venv/bin/python scripts/run_deppo_v2.py --device cuda --profile small --load .75 \
  --output outputs/deppo-v2-small --wandb-mode offline
```

入口先在验证集选择固定比例：0.1–0.9、步长0.1，分别结合 Random/Local/Forward。再对 DEPPO 和 MAPPO 各用种子0比较两种奖励×两档学习率(1e-4/3e-4)，每组65536步。按验证成功率优先、平均延迟次优选择配置，重新训练种子0/1/2各262144步。每8192步评估10个固定验证episode；最终30个独立测试episode不参与选择。

基线：Random 每请求 Bernoulli(.5) 转发、每周期比例 Uniform(.4,.6)；Always-local/forward 比例 .5；Tuned-fixed 用验证集选出的最优规则/比例；Queue-adaptive 用回源剩余字节/两方向剩余字节分配比例、无积压时 .5，使用与 Random 一样的随机转发及固定目标选择。

脚本保存任务签名与完成标记，重复运行跳过已完成任务；中断训练从 last.pt 恢复。改变配置必须换输出目录。输出 comparison.json/CSV、三种子波动及按训练种子和配对工作负载分层 bootstrap 的95%区间（2000次）。测试前核验所有方法工作负载一致。小样本区间仅用于描述不确定性，不保证 DEPPO 胜出。

## 论文—实现差异

| 项目 | 原文 | v2 |
|---|---|---|
| 带宽动作 | 回源/交付比例 | 恢复，限制 .05–.95 避免断流 |
| 带宽数值 | 12–36 Mbps | 原值保留审计；主实验按公开负载比校准 |
| 利用率公式 | 式(11)文字代入为闲置比例 | 修正为实际利用率 |
| 主奖励 | 成功率与利用率各 .5 | 成功/失败计数与小延迟惩罚，原风格作消融 |
| 数据 | Douyin 实际数据 | Poisson/Zipf 合成数据 |
| 网络 | TDD、FIFO 延迟公式 | SimGrid 流体容量动态更新、默认串行 FIFO |
| 历史网络细节 | 部分参数未说明 | 明确窗口8、GRU64、两层MLP128 |
| PPO | lr3e-4、clip.2 | 保守默认与等预算验证搜索，KL停止 |
| 对照 | RD/FO/DO/SAC/DD | RD/FO/DO、固定比例调优、自适应启发式、无历史MAPPO；不冒充DD/SAC |

本阶段不改变转发目标/本地调度目标算法，不实现监督预训练。

## 性能调优对照

可用 `--normalize-advantage` 开启逐 agent 优势归一化；`--initial-std .3` 缩小初始高斯探索标准差；`--hidden-size 256 --context-size 128` 扩大 actor/critic MLP 与 GRU。默认值不变，避免影响旧实验。所有选项写入配置和 checkpoint，续训必须保持一致。先在同场景/奖励/种子下比较优化器与探索配置，再单独比较网络容量，不能通过改负载把曲线变好。

确定性部署与随机策略诊断分别评估：`evaluate --exploration stochastic` 按固定评估种子采样模型动作，同时保护调用者的 PyTorch RNG 状态。默认仍是 deterministic；两个口径分别写入 evaluation.json，不能把随机策略诊断冒充默认验证成绩。参数化动作经过转发硬阈值，均值动作的收益不一定等于随机策略的期望收益。

奖励尺度对照使用 `configs/learning/small-reward-scaled.json`：`reward_scale=.1` 仅对返回给学习器的奖励乘正常数，不改变物理场景或业务目标。`train/business_reward`、`train/paper_reward` 保持未缩放，评估增加 `unscaled_episode_return`；跨尺度比较只能使用这些未缩放指标及成功率，不能把数值接近零解释为性能改善。此对照检验 critic 价值尺度，不预设它有效。

### Sampled-policy validation diagnostics

`train --eval-stochastic` additionally evaluates the sampled policy on exactly the
same fixed validation request seeds at each evaluation boundary. W&B/CSV uses
`eval_stochastic/*`, and per-episode results are saved separately as
`evaluation-stochastic-<env_steps>.json`. Existing `eval/*` curves and `best.pt`
selection remain deterministic. The option may be enabled when resuming an old
five-action checkpoint; it does not alter the optimizer or training configuration.
Evaluation preserves the training Torch RNG, and the sampled evaluation repeats
with a fixed policy RNG seed per request workload.

This distinction matters for DEPPO: taking the policy's deterministic continuous
parameters and then thresholding the forwarding score can behave very differently
from sampling parameters as during PPO collection. Compare both curves explicitly;
a good sampled score does not establish deterministic deployment performance.
