# Linux / RTX 4090 采样优化验收

2026-09-20，在原租用的 RTX 4090 / 16 vCPU 机器上验证。沿用项目 `.venv`（Python 3.11.16、Torch 2.9.1+cu128、SimGrid 4.1），没有重装 CUDA 或 PyTorch。

服务器原源码快照保留在 `/root/ecsai-before-git-20260920`；`/root/ecsai` 已改为真正的 Git 检出，分支 `codex/simgrid-framework-v1`。现有虚拟环境和旧实验结果仍在原项目路径使用。

## 同机性能对照

优化前源码 `3a695df`，优化后 `e25b8ae`。small 配置、种子 0、每 worker 32 周期、相同随机参数动作；各运行三遍，表中取中位数。启动时间另计；吞吐为环境步，不乘 agent 数。

| workers | 优化前 环境步/s | 优化后 环境步/s | 提升 | 优化后 已结束请求/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.42 | 35.26 | 14.59× | 911 |
| 2 | 4.70 | 63.19 | 13.45× | 1,587 |
| 4 | 10.72 | 120.47 | 11.24× | 3,161 |

4 workers 共 128 环境步，采样从 11.944 秒降至 1.062 秒；启动中位数由 3.799 秒降至 1.784 秒。按这两项中位数相加，包含启动约为 15.743 秒对 2.846 秒（约 5.53×）。

旧版本由 `git archive 3a695df packages` 生成，只为对照通过 `PYTHONPATH` 使用。其 benchmark 元信息中的 `code_revision` 来自运行目录，显示当前检出；实际源版本以本表、`summary.json` 中的 `benchmark_before_revision`、归档及源码指纹为准。

## CUDA 训练与恢复

DEPPO-adapted，small，4 workers，每 episode 128 周期，每批 512 环境步，10 轮更新、minibatch64，种子 0，W&B offline。先训练 4 episodes，再从 checkpoint 续训到总计 8 episodes。

| 阶段 | 新采样步数 | 采样墙钟 | 推理耗时（包含在采样中） | 更新耗时 |
| --- | ---: | ---: | ---: | ---: |
| 首批 | 512 | 8.80s | 0.49s | 2.28s |
| 续训批次 | 512 | 8.48s | 0.48s | 2.06s |

此前服务器首批采样记录为 51.60 秒；本次 8.80 秒约为其 1/5.9。该历史比较仅供参考，严格同机重复对照以上面的 benchmark 为准。实际训练包含 actor、TorchRL 收集以及 episode 重置，不能直接用短 benchmark 吞吐替代。

最初 GPU 续训暴露了随机数状态加载问题：`map_location="cuda"` 将 checkpoint 内 CUDA RNG 字节状态也移到显卡，而恢复接口要求 CPU ByteTensor。本机修复为恢复前显式 `.cpu()`，增加 CPU 合同测试和真实 CUDA checkpoint 续训测试，提交 `f22e2b1` 后由服务器 `git pull --ff-only` 获取；没有在服务器修改生产源码。

最终续训达到 **1,024 环境步**，36 个保存的参数张量发生更新，所有参数均为有限值，last/best checkpoint、评估结果和 W&B offline 文件均生成。仅两个训练批次、每次一个固定评估 episode，不据此判断算法优劣。

200ms 间隔 GPU 监控观测到最高显存约 532MiB、最高 GPU 利用率 10%；这是离散采样观测值，可能漏掉短峰值。小型网络及 CPU 采样仍使 GPU 使用呈间歇性，并未宣称已将 GPU 跑满。

## 回归与产物

- 本机修复验证：学习包 10 passed、1 skipped（本机无 NVIDIA，真实 CUDA 测试跳过）；Ruff 与 diff 检查通过。
- 修复前服务器全套测试：174 passed。
- 修复后服务器全套测试：**176 passed，39.14 秒**，包括真实 GPU 保存/恢复测试。
- 服务端及本地备份目录：`outputs/linux-optimized-validation/`。含 `before-*/`、`after-*/`、`cuda-small/`、`cuda-resume-fixed/`、`summary.json`、pytest 日志及 GPU CSV；首次失败的 `cuda-resume/` 日志也保留。
- 本轮全部使用 W&B offline，没有验证 online。训练和 GPU 监控进程均已结束，未启动 2,048 episode 正式实验，也未关闭租用机器。

后续更新使用：

```sh
cd /root/ecsai
git pull --ff-only
.venv/bin/python -m edge_sim_learning.cli benchmark --profile small \
  --workers 1 2 4 --steps 32 --seed 0 --wandb-mode offline \
  --output outputs/my-benchmark
```
