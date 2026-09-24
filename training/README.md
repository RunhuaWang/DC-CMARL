# Training stage

本目录复现论文采用的 6×6 dual-conditioned MAPPO 训练阶段。

## 方法与设置

- 4 个独立 Actor，输入包含固定责任区 owner flags；
- Actor 采用连续 FiLM dual conditioning；
- centralized reward/cost differential critics 采用连续 dual 直接条件化；
- 21 个训练 dual：`0, 0.5, ..., 10`；
- 42 个并行环境，每个 dual 对应 2 个环境；
- rollout length 128，differential horizon 32；
- per-dual advantage normalization、分层 minibatch、PCGrad；
- per-dual adaptive KL retention，每个 PPO minibatch 生效；
- seed 0，共 22,579,200 joint environment steps；
- 正式评价使用 stochastic action sampling 和 seeds 1000--1019。

## 安装与运行

在本目录执行：

```bash
python -m venv .venv
.venv/bin/pip install -e '.[train]'
.venv/bin/python src/hrmr/experiments/fixed_responsibility_small_6x6_dual_conditioned_mappo/run_dual_conditioned_film_full_grid_pcgrad_adaptive_retention.py
```

训练会写入：

```text
map_6x6/data/dual_conditioned/film_actor_pcgrad_adaptive_retention_full_grid_21_lambda_seed0/
map_6x6/images/dual_conditioned/film_actor_pcgrad_adaptive_retention_full_grid_21_lambda_seed0/
```

## 已包含结果

- `best.pt`：执行阶段使用的最终 checkpoint；
- `map_6x6/images/paper/fixed_responsibility_6x6_scenario.{png,pdf}`：论文场景图。

为保持投稿仓库精简，本副本不包含训练曲线、逐 dual evaluation、轨迹图或其他
诊断图片；重新训练时这些文件仍会由代码正常生成。
