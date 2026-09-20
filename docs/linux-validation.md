# Linux / RTX 4090 验证记录

2026-09-20 在租用机器上验证。服务器项目目录为 `/root/ecsai`；源码快照包含当时本地未提交修改，校验清单保存在服务器 `SOURCE_SNAPSHOT.json`。未启动完整 2048 episode 训练。

## 环境

- Ubuntu 22.04，RTX 4090 24GB，驱动 595.80。
- 16 vCPU，系统呈现 8 核 / 16 线程，型号 AMD EPYC 7543；不据此推断物理核独占。
- 系统可见内存约 92GiB，容器 memory.max 约 86GiB；安装后磁盘剩余约 68GB。
- 预装 CUDA Toolkit 13.2；Conda `/usr/local/miniconda3/envs/py312` 中已有 Python 3.12.11、Torch 2.13.0+cu132，CUDA 可用，未修改该环境。
- 项目独立 `.venv` 使用 Python 3.11.16、Torch 2.9.1+cu128、TorchRL 0.10.1、BenchMARL 1.5.2 与锁定依赖。没有重装显卡驱动或系统 CUDA Toolkit。
- SimGrid 4.1 从锁定源包编译并验证，wheel 在服务器 `dist/simgrid/`。

最初只检查了 SSH 非交互默认 Python，漏查 Conda 预装环境；用户指出后已核实上述预装版本。项目独立安装是为保持锁文件版本，预装版本与项目版本不同。

## 验证结果

- Linux 全套测试：168 passed，30.67 秒。
- CUDA HistoryActor 前向/反向通过，GRU 梯度非零；单 actor 43464 参数。
- BenchMARL CUDA smoke：4 episodes × 8 cycles，2 workers，32 环境步，完成采样、更新、评估与 checkpoint。
- small 场景 CUDA 验证：4 episodes × 128 cycles，4 workers，512 环境步、10 轮更新、minibatch64，完成评估与 checkpoint。
- 本次全部使用 W&B offline；online 未验证。

small 验证一批的采样耗时约 **51.60s**，参数更新阶段约 **1.59s**。1 秒间隔的 nvidia-smi 采样中，最高观测显存约 **530MiB**，最高观测 GPU 利用率 13%；这不是连续采样的瞬时峰值。当前瓶颈主要在仿真和采样，不能靠升级 5090 直接解决。

## worker 吞吐

small 配置，每 worker 32 周期，随机参数动作，启动时间另计。不同 worker 数包含不同数量的独立工作负载，单次测量仅用于选初始配置，不代表严格线性扩展系数。

| workers | 环境步/s | 已结束请求/s | 采样墙钟(s) | 初次启动(s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.71 | 69.91 | 11.83 | 0.93 |
| 2 | 5.07 | 127.23 | 12.63 | 1.66 |
| 4 | 12.79 | 335.73 | 10.01 | 2.96 |

本机先使用默认 4 workers。该测试与先前 macOS 的短 episode 测试条件不同，不直接比较硬件优劣。

## 运行与产物

```sh
cd /root/ecsai
source .venv/bin/activate

# 重新运行 CUDA smoke
edge-learn train --profile smoke --output outputs/my-cuda-smoke \
  --episodes 4 --workers 2 --batch 32 --epochs 1 --minibatch 16 \
  --device cuda --eval-interval 32 --eval-episodes 1 --wandb-mode offline
```

服务器原始产物：`/root/ecsai/outputs/linux-validation/`，包含 CUDA 信息、pytest 日志、两次训练的 CSV/checkpoint/评估/W&B offline 文件、benchmark 和 GPU 采样日志。

本地已备份到 `outputs/rental-linux/outputs/linux-validation/`。完整实验参数与三种子运行方式参见 [DEPPO 说明](deppo.md)。验证完成后训练进程与临时 GPU 监控均已结束。
