# 6×6 fixed-responsibility dual-conditioned MAPPO

本目录只保留当前采用的 dual-conditioned 训练方法。它与 fixed-λ trainer 隔离，
不会加载或覆盖 fixed-λ 结果。

## 当前正式方法

- 6×6 固定责任区环境，owner flag 保留；
- 四个参数独立的 actors；
- Actor 使用连续 FiLM dual conditioning；
- centralized reward/cost differential critics 使用直接拼接的连续 dual；
- 训练网格为 `0, 0.5, ..., 10`，共 21 个 dual；
- 42 个并行环境，每个 dual 固定绑定 2 个环境；
- rollout length 为 128，n-step differential horizon 为 32；
- 每个 dual 独立维护 reward/cost average rate，并按 dual 标准化 advantage；
- PPO minibatch 按 dual 分层；
- Actor 使用 per-λ PCGrad 缓解梯度冲突；
- 每个 dual 使用独立的自适应 KL retention coefficient；
- retention loss 在每个 PPO minibatch 生效；
- 不读取解析模式、理论切换点或目标 occupancy 标签。

训练共 4,200 次 rollout/update，即 22,579,200 joint environment steps；每个
dual 获得 1,075,200 条 transition。正式评价采用 stochastic action sampling。

## 直接运行

在 IDE 中运行：

```text
run_dual_conditioned_film_full_grid_pcgrad_adaptive_retention.py
```

也可仅检查配置，不启动训练：

```bash
.venv/bin/python -m hrmr.experiments.fixed_responsibility_small_6x6_dual_conditioned_mappo.run_dual_conditioned_film_full_grid_pcgrad_adaptive_retention --check-only
```

唯一配置文件：

```text
configs/dual_conditioned_film_full_grid_pcgrad_adaptive_retention_seed0.toml
```

结果目录：

```text
map_6x6/data/dual_conditioned/film_actor_pcgrad_adaptive_retention_full_grid_21_lambda_seed0/
map_6x6/images/dual_conditioned/film_actor_pcgrad_adaptive_retention_full_grid_21_lambda_seed0/
```

上述目录是同一方法、同一 seed 的固定输出槽位；再次运行会按现有事务式覆盖机制
更新它们，不会写入旧实验名称。
