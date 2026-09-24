# ICASSP HRMR 6×6 reproducibility package

本目录是从完整研究仓库中整理出的投稿版本，只包含论文采用的 6×6 固定责任区
实验主线

目录分为两个相互独立、可安装运行的阶段：

- `training/`：训练连续 dual-conditioned MAPPO 策略；
- `execution/`：冻结训练策略，运行无中心对偶更新并生成论文结果。

两个阶段均包含各自所需的 `src/`、配置、结果和图片。请在对应阶段目录中创建
Python 环境并执行命令，避免输出路径相互混用。

## 论文主线

训练阶段使用 6×6 固定责任区环境、连续 FiLM Actor、集中式 reward/cost critics、
per-dual PCGrad 与自适应 KL retention。训练 dual 网格为
`0, 0.5, ..., 10`。

执行阶段冻结 Actor，不再更新神经网络；机器人通过环形通信聚合全局 cost，并从
`lambda_0=0` 开始执行投影对偶迭代。论文汇总采用 `H=64`、1024 次对偶更新，
约束阈值为 `0, 0.1, 0.2, 0.4, 0.6, 0.9, 1.0, 1.2`。

更具体的运行方式见两个子目录中的 README。

