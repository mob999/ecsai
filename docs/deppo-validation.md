# DEPPO 接入验收记录

本记录来自 macOS ARM64 CPU，2026-09-20。未执行 Linux CUDA 正式训练或 W&B online。W&B offline 已验证，六个学习运行、三种启发式和一个 benchmark 共生成 10 个离线运行文件。

## 验证

- 全套 `pytest`：168 passed。后续 checkpoint 最佳模型恢复调整的学习专项：6 passed。
- Ruff、`git diff --check`、`uv lock --check` 通过。
- SDK 单独导入不加载 NumPy、Torch、PettingZoo 或 BenchMARL。
- PettingZoo Parallel API、TorchRL spec、截断 bootstrap、历史清空/有效长度、GRU 梯度、独立 actor、共享 critic、CPU 更新和 checkpoint 恢复均通过。
- 原 SDK 测试继续通过；新测试检查独立/共享链路、并发 FIFO、回源合并与独立交付、溢出、取消、LRU、精确截止和同刻释放容量。

## 小预算对照

使用 small 场景（3 集群、10 缓存、300 请求/s），为快速验收将每 episode 缩至 16 周期。两种学习方法分别训练 seed 0/1/2，每种子仅 8 episodes = 128 环境步，2 workers，batch64、minibatch64、10 次更新轮次。其余采用默认参数。每个方法都用相同的 10 个固定评估 episode；脚本断言实际 workload 元信息完全一致。

这只是学习和评估闭环验证，不能用于论文性能结论；正式默认是 2048 × 128 环境步。延迟只统计成功请求，所有方法在截断后继续排空。

| 方法 | 成功率 | 成功请求平均延迟 (s) | 训练种子成功率 |
| --- | ---: | ---: | --- |
| DEPPO-adapted | 97.27% | 0.3519 | 91.84%, 99.98%, 100.00% |
| MAPPO-no-context | 99.86% | 0.2794 | 99.92%, 99.65%, 100.00% |
| random | 100.00% | 0.2546 | 100.00% |
| local | 99.65% | 0.2829 | 99.65% |
| forward | 99.98% | 0.2801 | 99.98% |

学习方法的表内结果为三训练种子的平均，每个种子先对相同 10 个评估 episode 求平均。启发式没有训练种子，表内数值来自同一组评估 episode。DEPPO 在这个预算下没有稳定优于对照，不据此调整场景负载或筛选种子。

## 采样性能

同一 small 场景、16 周期 episode，随机参数动作；每 worker 执行 64 周期。吞吐包含 episode 结束后的物理进程重建，启动计时单独列出。

| workers | 环境步/s | 已结束请求/s | 采样墙钟 (s) | 初次启动 (s) | 仿真 worker-seconds | IPC/等待 worker-seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16.993 | 395.888 | 3.766 | 0.223 | 3.181 | 0.046 |
| 2 | 22.927 | 534.132 | 5.583 | 0.429 | 7.940 | 1.149 |
| 4 | 22.299 | 519.324 | 11.480 | 0.900 | 32.919 | 4.850 |

本机 4 workers 没有比 2 workers 更快，不能据此假定线性扩展或宣称已解决 GPU 利用率问题。原生仿真仍是主要耗时；短 episode 还承担进程重建成本。CPU smoke 的神经网络很小，CUDA 吞吐应在目标机器重新测量。

## 复现与产物

```sh
uv run --package edge-sim-learning python scripts/validate_learning.py \
  --output outputs/deppo-validation-release
```

原始产物保存在 `outputs/deppo-validation-release/`，包括 `comparison.csv/json`、每运行的配置/CSV/W&B offline、学习 checkpoint、10-episode 评估明细，以及 `benchmark-small/benchmark.json`。该目录被 git 忽略；本记录及场景配置保留在源码中。完整用法、模型定义和论文差异见 [实验说明](deppo.md)。
