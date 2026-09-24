"""6×6 fixed-responsibility dual-conditioned average-reward MAPPO 训练入口。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hrmr.experiments.assigned_targets_ppo.trainer import (
    physical_occupancy_diagnostics,
    write_json,
    write_rows,
)
from hrmr.experiments.assigned_targets_random_restart_ppo.sampling import RestartSchedule
from hrmr.experiments.fixed_responsibility_small_6x6_mappo.sampling import (
    SmallSixBySixRandomRestartVectorEnv,
)
from hrmr.rendering import render_layout, render_trajectory
from hrmr.rl.device import resolve_device, seed_everything
from hrmr.rl.fixed_lambda_trainer import _initial_occupancies, _make_actor_optimizer
from hrmr.rl.target_permutation import sample_target_permutations

from .advantages import mixed_conditioned_restart_quantities
from .config import (
    CONFIG_PATH,
    DualConditionedConfig,
    default_image_output,
    default_output,
    load_config,
    load_environment,
)
from .evaluation import evaluate_dual_conditioned_actor
from .networks import make_dual_conditioned_networks
from .policy_retention import PerLambdaPolicyRetention
from .sampling import collect_dual_conditioned_restart_rollout
from .schedule import ConditionalAverageRateBank, PersistentDualBatchSchedule
from .theory import MODE_LABELS, all_mode_rates, analytic_reference, optimal_mode_mask
from .update import update_models

CALIBRATION_SEEDS = (2000, 2001, 2002, 2003, 2004)
CALIBRATION_BURN_IN = 100
CALIBRATION_STEPS = 500
CHECKPOINT_FORMAT = "fixed_responsibility_small_6x6_dual_conditioned_mappo_v1"
PCGRAD_RNG_DOMAIN = 0x50434752
RETENTION_RNG_DOMAIN = 0x5245544E


def _actor_initialization_seeds(spec: DualConditionedConfig) -> tuple[int, ...]:
    base = spec.base.experiment.independent_actor_init_seed_base + 4 * spec.training_seed
    return tuple(base + index for index in range(4))


def _make_dual_schedule(spec: DualConditionedConfig):
    return PersistentDualBatchSchedule(
        spec.dual_values,
        spec.base.training.num_parallel_envs,
        spec.training_seed,
    )


def _source_fingerprint() -> str:
    source_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _nonempty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def _stage_existing(root: Path, image_root: Path, spec: DualConditionedConfig):
    """仅允许相同 dual-conditioned 槽位事务式覆盖。"""

    paths = (root, image_root)
    for path in paths:
        if not _nonempty(path):
            continue
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"refusing to overwrite unrecognized output: {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("method_family") != "dual_conditioned"
            or manifest.get("experiment") != spec.base.experiment.profile
            or manifest.get("training_seed") != spec.training_seed
        ):
            raise FileExistsError("existing output belongs to another experiment")
        if manifest.get("status") == "running":
            raise FileExistsError(
                "existing output is marked running; refusing concurrent overwrite"
            )
    staged = []
    for path in paths:
        if not _nonempty(path):
            continue
        backup = path.with_name(f".{path.name}.overwrite-backup")
        if backup.exists():
            raise FileExistsError(f"stale overwrite backup exists: {backup}")
        path.replace(backup)
        staged.append((path, backup))
    return staged


def _restore_staged(staged):
    for original, backup in reversed(staged):
        if original.exists():
            shutil.rmtree(original) if original.is_dir() else original.unlink()
        backup.replace(original)


def _discard_staged(staged):
    for _, backup in staged:
        shutil.rmtree(backup) if backup.is_dir() else backup.unlink()


def _training_rows(
    batch,
    physical_membership,
    dual_assignments,
    spec,
    steps,
    update,
    rates,
    statistics,
    elapsed,
):
    rows = []
    reward_rates = dict(zip(spec.dual_values, rates.reward_rates, strict=True))
    cost_rates = dict(zip(spec.dual_values, rates.cost_rates, strict=True))
    target_ids = tuple(("S1", "S2", "S3", "S4", "H1", "H2", "H3", "H4"))
    for dual in spec.dual_values:
        environment_mask = np.asarray(dual_assignments) == dual
        rewards = batch.rewards[:, environment_mask]
        costs = batch.global_costs[:, environment_mask]
        occupancies = batch.next_occupancies[:, environment_mask]
        modes = all_mode_rates(occupancies, target_ids)
        optimal = float(np.mean(optimal_mode_mask(occupancies, target_ids, fixed_lambda=dual)))
        physical = physical_occupancy_diagnostics(physical_membership[:, environment_mask])
        mean_off = float(np.mean([physical[f"non_target_agent_{index}"] for index in range(1, 5)]))
        mean_duplicate = float(
            np.mean([physical[f"duplicate_target_agent_{index}"] for index in range(1, 5)])
        )
        row = {
            "update": update,
            "environment_steps": steps,
            "dual_lambda": dual,
            "normalized_dual": dual / spec.conditioning_max,
            "num_parallel_envs": int(np.sum(environment_mask)),
            "rollout_mean_reward": float(np.mean(rewards)),
            "rollout_mean_cost": float(np.mean(costs)),
            "rollout_scalarized_objective": float(np.mean(rewards - dual * costs)),
            "avg_reward_estimate": reward_rates[dual],
            "avg_cost_estimate": cost_rates[dual],
            "mean_num_distinct_targets": float(np.mean(np.sum(occupancies, axis=-1))),
            "all_safe_rate": float(np.mean(np.all(occupancies[..., :4] == 1, axis=-1))),
            "all_hazardous_rate": float(np.mean(np.all(occupancies[..., 4:] == 1, axis=-1))),
            "optimal_mode_rate": optimal,
            "mean_off_target_rate": mean_off,
            "mean_duplicate_target_rate": mean_duplicate,
            "entropy": statistics.entropy,
            "reward_critic_loss": statistics.reward_critic_loss,
            "cost_critic_loss": statistics.cost_critic_loss,
            "actor_loss": statistics.actor_loss,
            "actor_gradient_aggregation": statistics.actor_gradient_aggregation,
            "pcgrad_conflict_fraction": statistics.pcgrad_conflict_fraction,
            "pcgrad_mean_pairwise_cosine": statistics.pcgrad_mean_pairwise_cosine,
            "pcgrad_projection_relative_change": (statistics.pcgrad_projection_relative_change),
            "policy_retention_mode": spec.policy_retention,
            "policy_retention_active": statistics.policy_retention_active,
            "policy_retention_coefficient": statistics.policy_retention_coefficient_by_dual[
                spec.dual_values.index(dual)
            ],
            "policy_retention_next_coefficient": (
                statistics.policy_retention_next_coefficient_by_dual[spec.dual_values.index(dual)]
            ),
            "policy_retention_target_kl": spec.policy_retention_target_kl,
            "policy_retention_evaluations_per_dual": (
                statistics.policy_retention_evaluations_per_dual
            ),
            "policy_retention_kl": statistics.policy_retention_kl,
            "policy_retention_loss": statistics.policy_retention_loss,
            "policy_retention_kl_for_lambda": statistics.policy_retention_kl_by_dual[
                spec.dual_values.index(dual)
            ],
            **{
                f"pcgrad_conflict_fraction_actor_{index + 1}": value
                for index, value in enumerate(statistics.pcgrad_conflict_fraction_by_agent)
            },
            **{
                f"pcgrad_mean_pairwise_cosine_actor_{index + 1}": value
                for index, value in enumerate(statistics.pcgrad_mean_pairwise_cosine_by_agent)
            },
            **{
                f"pcgrad_projection_relative_change_actor_{index + 1}": value
                for index, value in enumerate(statistics.pcgrad_projection_relative_change_by_agent)
            },
            "elapsed_seconds": elapsed,
            **{f"mode_{label}_rate": modes[label] for label in MODE_LABELS},
            **{
                f"occupancy_{target}": float(value)
                for target, value in zip(target_ids, np.mean(occupancies, axis=(0, 1)), strict=True)
            },
        }
        rows.append(row)
    return rows


def _evaluation_row(evaluation, payload, dual, step, initial_mode, order):
    aggregate = evaluation.aggregate
    physical = payload["agent_occupancy"]
    reference = analytic_reference(dual)
    modes = payload["aggregate"]["steady_operating_mode_rates"]
    objective = aggregate.steady_mean_scalarized_objective
    return {
        "environment_steps": step,
        "dual_lambda": dual,
        "normalized_dual": dual / 10.0,
        "initial_state_mode": initial_mode,
        "target_order": order,
        "reward": aggregate.steady_mean_reward,
        "cost": aggregate.steady_mean_cost,
        "objective": objective,
        "optimal_objective": reference.scalarized_objective,
        "objective_gap": reference.scalarized_objective - objective,
        "optimal_mode_rate": aggregate.steady_mean_operating_mode_rate,
        "all_safe_rate": payload["aggregate"]["steady_owner_valid_all_safe_mode_rate"],
        "all_hazardous_rate": aggregate.steady_mean_all_hazardous_mode_rate,
        "valid_distinct_targets": aggregate.steady_mean_num_distinct_targets,
        "duplicate_rate": float(np.mean(physical["duplicate_target_rate"])),
        "off_target_rate": float(np.mean(physical["non_target_rate"])),
        "entropy": aggregate.steady_mean_actor_entropy,
        **{f"mode_{label}_rate": modes[label] for label in MODE_LABELS},
        **{
            f"occupancy_{target}": float(value)
            for target, value in zip(
                evaluation.target_ids,
                aggregate.steady_mean_target_occupancy,
                strict=True,
            )
        },
    }


def _evaluate_grid(
    actor,
    environment,
    spec,
    device,
    *,
    seeds,
    burn_in_steps,
    evaluation_steps,
    initial_mode,
    order,
    step,
    save_root=None,
):
    rows = []
    trajectories = {}
    for dual in spec.dual_values:
        evaluation, payload = evaluate_dual_conditioned_actor(
            actor,
            environment,
            dual,
            spec.conditioning_max,
            seeds,
            initial_state_mode=initial_mode,
            target_order_mode=order,
            evaluation_steps=evaluation_steps,
            burn_in_steps=burn_in_steps,
            device=device,
        )
        row = _evaluation_row(evaluation, payload, dual, step, initial_mode, order)
        rows.append(row)
        trajectories[dual] = evaluation.trajectories[0]
        if save_root is not None:
            slug = str(int(dual)) if float(dual).is_integer() else str(dual).replace(".", "p")
            destination = save_root / f"lambda{slug}"
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / f"{initial_mode}_{order}.json", payload)
    return rows, trajectories


def _print_grid(rows, *, label):
    print(f"[{label}] {len(rows)}-lambda summary", flush=True)
    for row in rows:
        modes = " ".join(f"{name}={row[f'mode_{name}_rate']:.2f}" for name in MODE_LABELS)
        print(
            f"  lambda={row['dual_lambda']:>4.1f} R={row['reward']:.4f} "
            f"C={row['cost']:.4f} F={row['objective']:.4f} "
            f"optimal={row['optimal_mode_rate']:.3f} distinct={row['valid_distinct_targets']:.3f} "
            f"off={row['off_target_rate']:.3f} dup={row['duplicate_rate']:.3f} | {modes}",
            flush=True,
        )


def _checkpoint(
    spec,
    actor,
    reward_critic,
    cost_critic,
    actor_optimizer,
    reward_optimizer,
    cost_optimizer,
    reward_rates,
    cost_rates,
    dual_schedule,
    restart_schedule,
    vector,
    permutation_rng,
    reset_rng,
    pcgrad_generator,
    policy_retention,
    steps,
    updates,
    calibration_score,
):
    return {
        "format": CHECKPOINT_FORMAT,
        "config": spec.base.to_dict(),
        "dual_values": list(spec.dual_values),
        "conditioning_max": spec.conditioning_max,
        "actor_conditioning": spec.actor_conditioning,
        "critic_conditioning": spec.critic_conditioning,
        "dual_assignment": spec.dual_assignment,
        "actor_gradient_aggregation": spec.actor_gradient_aggregation,
        "policy_retention": spec.policy_retention,
        "policy_retention_coefficient": spec.policy_retention_coefficient,
        "policy_retention_anchor_capacity": spec.policy_retention_anchor_capacity,
        "policy_retention_improvement_tolerance": (spec.policy_retention_improvement_tolerance),
        "policy_retention_apply_every_minibatch": (spec.policy_retention_apply_every_minibatch),
        "policy_retention_target_kl": spec.policy_retention_target_kl,
        "policy_retention_adaptation_rate": spec.policy_retention_adaptation_rate,
        "policy_retention_min_coefficient": spec.policy_retention_min_coefficient,
        "policy_retention_max_coefficient": spec.policy_retention_max_coefficient,
        "training_seed": spec.training_seed,
        "actor_initialization_seeds": list(actor.initialization_seeds),
        "environment_steps": steps,
        "updates": updates,
        "calibration_score": calibration_score,
        "actor": actor.state_dict(),
        "reward_critic": reward_critic.state_dict(),
        "cost_critic": cost_critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "reward_optimizer": reward_optimizer.state_dict(),
        "cost_optimizer": cost_optimizer.state_dict(),
        "rho_reward_by_lambda": reward_rates.state_dict(),
        "rho_cost_by_lambda": cost_rates.state_dict(),
        "dual_schedule": dual_schedule.state_dict(),
        "restart_schedule": asdict(restart_schedule),
        "vector_env": vector.state_dict(),
        "permutation_rng_state": permutation_rng.bit_generator.state,
        "reset_rng_state": reset_rng.bit_generator.state,
        "pcgrad_rng_state": pcgrad_generator.get_state(),
        "policy_retention_state": (
            None if policy_retention is None else policy_retention.state_dict()
        ),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def _plot_final(rows, image_root):
    selected = [
        row
        for row in rows
        if row["initial_state_mode"] == "random" and row["target_order"] == "canonical"
    ]
    lambdas = np.asarray([row["dual_lambda"] for row in selected])
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(lambdas, [row["reward"] for row in selected], marker="o")
    axes[0].set_ylabel("Steady reward")
    axes[0].grid(alpha=0.3)
    axes[1].plot(lambdas, [row["cost"] for row in selected], marker="o", color="tab:red")
    axes[1].set_xlabel("Dual lambda")
    axes[1].set_ylabel("Steady global cost")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(image_root / "reward_cost_by_lambda.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for mode in MODE_LABELS:
        axis.plot(lambdas, [row[f"mode_{mode}_rate"] for row in selected], label=mode)
    axis.set(xlabel="Dual lambda", ylabel="Steady mode rate", ylim=(-0.02, 1.02))
    axis.grid(alpha=0.3)
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(image_root / "mode_rates_by_lambda.png", dpi=180)
    plt.close(figure)


def train(
    config_path=CONFIG_PATH,
    output_root=None,
    *,
    device_override="cpu",
    overwrite_existing=True,
) -> Path:
    """冷启动一个 seed=0 mixed-dual run；不会读取 fixed-λ checkpoints。"""

    spec = load_config(config_path)
    environment = load_environment(spec)
    root = Path(output_root or default_output(spec)).resolve()
    image_root = Path(
        default_image_output(spec) if output_root is None else root / "images"
    ).resolve()
    device = resolve_device(device_override or spec.base.experiment.device)
    if (_nonempty(root) or _nonempty(image_root)) and not overwrite_existing:
        raise FileExistsError(f"refusing to overwrite non-empty output: {root}")
    staged = (
        _stage_existing(root, image_root, spec) if _nonempty(root) or _nonempty(image_root) else []
    )
    manifest = {
        "experiment": spec.base.experiment.profile,
        "method_family": "dual_conditioned",
        "status": "running",
        "training_seed": spec.training_seed,
        "dual_values": list(spec.dual_values),
        "conditioning_max": spec.conditioning_max,
        "actor_conditioning": spec.actor_conditioning,
        "critic_conditioning": spec.critic_conditioning,
        "actor_gradient_aggregation": spec.actor_gradient_aggregation,
        "policy_retention": spec.policy_retention,
        "policy_retention_coefficient": spec.policy_retention_coefficient,
        "policy_retention_anchor_capacity": spec.policy_retention_anchor_capacity,
        "policy_retention_improvement_tolerance": (spec.policy_retention_improvement_tolerance),
        "policy_retention_apply_every_minibatch": (spec.policy_retention_apply_every_minibatch),
        "policy_retention_target_kl": spec.policy_retention_target_kl,
        "policy_retention_adaptation_rate": spec.policy_retention_adaptation_rate,
        "policy_retention_min_coefficient": spec.policy_retention_min_coefficient,
        "policy_retention_max_coefficient": spec.policy_retention_max_coefficient,
        "environments_per_dual": spec.environments_per_dual,
        "progress_every_updates": spec.progress_every_updates,
        "dual_assignment": spec.dual_assignment,
        "config": spec.base.to_dict(),
        "checkpoint_format": CHECKPOINT_FORMAT,
        "code_fingerprint": _source_fingerprint(),
        "environment_steps": 0,
        "updates": 0,
        "data_output": str(root),
        "image_output": str(image_root),
    }
    try:
        root.mkdir(parents=True, exist_ok=True)
        image_root.mkdir(parents=True, exist_ok=True)
        write_json(root / "run_manifest.json", manifest)
        source = Path(spec.base.source_path)
        shutil.copy2(source, root / "config.toml")
        shutil.copy2(
            source.parent / spec.base.experiment.environment_config,
            root / "environment.toml",
        )
        _run(spec, environment, device, root, image_root, manifest)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        write_json(root / "run_manifest.json", manifest)
        _restore_staged(staged)
        raise
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        write_json(root / "run_manifest.json", manifest)
        _restore_staged(staged)
        raise
    _discard_staged(staged)
    return root


def _run(spec, environment, device, root, image_root, manifest):
    config = spec.base
    settings = config.training
    render_layout(environment, image_root / "layout.png", title="6x6 dual-conditioned MAPPO")
    torch.set_num_threads(config.experiment.torch_num_threads)
    seed_everything(spec.training_seed)
    actor, reward_critic, cost_critic = make_dual_conditioned_networks(
        _actor_initialization_seeds(spec),
        activation=config.experiment.activation,
        actor_conditioning=spec.actor_conditioning,
        critic_conditioning=spec.critic_conditioning,
        device=device,
    )
    reward_optimizer = torch.optim.Adam(
        reward_critic.parameters(), lr=settings.critic_learning_rate
    )
    cost_optimizer = torch.optim.Adam(cost_critic.parameters(), lr=settings.critic_learning_rate)
    reward_rates = ConditionalAverageRateBank(spec.dual_values, settings.avg_rate_ema)
    cost_rates = ConditionalAverageRateBank(spec.dual_values, settings.avg_rate_ema)
    dual_schedule = _make_dual_schedule(spec)
    permutation_rng = np.random.default_rng(
        np.random.SeedSequence((spec.training_seed, 0x54415247))
    )
    reset_rng = np.random.default_rng(
        np.random.SeedSequence((spec.training_seed, spec.reset_rng_domain))
    )
    pcgrad_generator = torch.Generator(device="cpu")
    pcgrad_generator.manual_seed(PCGRAD_RNG_DOMAIN + spec.training_seed)
    policy_retention = PerLambdaPolicyRetention(
        spec.dual_values,
        num_agents=4,
        observation_dim=57,
        anchor_capacity=spec.policy_retention_anchor_capacity,
        improvement_tolerance=spec.policy_retention_improvement_tolerance,
        seed=RETENTION_RNG_DOMAIN + spec.training_seed,
        device=device,
        adaptive_initial_coefficient=spec.policy_retention_coefficient,
        adaptive_target_kl=spec.policy_retention_target_kl,
        adaptive_rate=spec.policy_retention_adaptation_rate,
        adaptive_min_coefficient=spec.policy_retention_min_coefficient,
        adaptive_max_coefficient=spec.policy_retention_max_coefficient,
    )
    actor_optimizer = _make_actor_optimizer(actor, settings.actor_learning_rate)
    vector = SmallSixBySixRandomRestartVectorEnv(settings.num_parallel_envs, environment)
    restart_schedule = RestartSchedule(interval=spec.restart_interval_per_environment)
    observations, states, infos = vector.reset_random(reset_rng)
    occupancies = _initial_occupancies(infos)
    steps = updates = 0
    next_calibration = settings.evaluation_interval
    best_score = -float("inf")
    best_step = 0
    calibration_rows = []
    started = time.perf_counter()
    print(
        f"[START] dual-conditioned seed={spec.training_seed} duals={spec.dual_values} "
        f"actor={spec.actor_conditioning} "
        f"critics={spec.critic_conditioning} "
        f"actor_gradients={spec.actor_gradient_aggregation} "
        f"retention={spec.policy_retention} "
        f"assignment={spec.dual_assignment} "
        f"envs={settings.num_parallel_envs} rollout={settings.rollout_length} "
        f"n={settings.n_step} steps={settings.total_environment_steps}",
        flush=True,
    )

    training_fields = None
    with (root / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        while steps < settings.total_environment_steps:
            dual_assignments = dual_schedule.next()
            permutations = sample_target_permutations(
                permutation_rng,
                num_permutations=vector.num_envs,
                num_targets=8,
            )
            rollout = collect_dual_conditioned_restart_rollout(
                actor,
                vector,
                observations,
                states,
                occupancies,
                settings.rollout_length,
                device,
                permutations,
                dual_lambda=dual_assignments,
                conditioning_max=spec.conditioning_max,
                schedule=restart_schedule,
                reset_rng=reset_rng,
                rollout_start_environment_steps=steps,
            )
            observations, states, occupancies = (
                rollout.observations,
                rollout.states,
                rollout.occupancies,
            )
            steps += rollout.batch.num_steps * rollout.batch.num_envs
            updates += 1
            policy_retention.observe(rollout.batch.observations, dual_assignments)
            frozen = mixed_conditioned_restart_quantities(
                rollout,
                reward_critic,
                cost_critic,
                reward_rates,
                cost_rates,
                dual_assignments,
                settings.normalize_combined_advantage,
                device,
                settings.n_step,
            )
            statistics = update_models(
                rollout.batch,
                dual_assignments,
                actor,
                reward_critic,
                cost_critic,
                actor_optimizer,
                reward_optimizer,
                cost_optimizer,
                frozen.reward_targets,
                frozen.cost_targets,
                frozen.combined_advantages,
                config,
                device,
                actor_gradient_aggregation=spec.actor_gradient_aggregation,
                pcgrad_generator=pcgrad_generator,
                policy_retention=policy_retention,
                policy_retention_coefficient=spec.policy_retention_coefficient,
                policy_retention_apply_every_minibatch=(
                    spec.policy_retention_apply_every_minibatch
                ),
            )
            rows = _training_rows(
                rollout.batch,
                rollout.physical_membership,
                dual_assignments,
                spec,
                steps,
                updates,
                frozen,
                statistics,
                time.perf_counter() - started,
            )
            if writer is None:
                training_fields = tuple(rows[0])
                writer = csv.DictWriter(stream, fieldnames=training_fields)
                writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            manifest.update(environment_steps=steps, updates=updates)

            if updates == 1 or updates % spec.progress_every_updates == 0:
                anchors = {0.0, 2.5, 5.0, 7.0, 9.0, 10.0}
                next_coefficients = np.asarray(
                    statistics.policy_retention_next_coefficient_by_dual,
                    dtype=np.float64,
                )
                finite_coefficients = next_coefficients[np.isfinite(next_coefficients)]
                retention_beta = (
                    "inactive"
                    if finite_coefficients.size == 0
                    else f"[{np.min(finite_coefficients):.3g},{np.max(finite_coefficients):.3g}]"
                )
                summary = " | ".join(
                    f"l={row['dual_lambda']:g}:R={row['rollout_mean_reward']:.2f},"
                    f"C={row['rollout_mean_cost']:.2f},opt={row['optimal_mode_rate']:.2f}"
                    for row in rows
                    if row["dual_lambda"] in anchors
                )
                print(
                    f"[TRAIN] update={updates} step={steps}/{settings.total_environment_steps} "
                    f"entropy={statistics.entropy:.3f} "
                    f"conflict={statistics.pcgrad_conflict_fraction:.3f} "
                    f"cos={statistics.pcgrad_mean_pairwise_cosine:.3f} "
                    f"retention_kl={statistics.policy_retention_kl:.5f} "
                    f"retention_beta={retention_beta} "
                    f"elapsed={time.perf_counter() - started:.0f}s | "
                    f"{summary}",
                    flush=True,
                )

            if steps >= next_calibration or steps == settings.total_environment_steps:
                measured, _ = _evaluate_grid(
                    actor,
                    environment,
                    spec,
                    device,
                    seeds=CALIBRATION_SEEDS,
                    burn_in_steps=CALIBRATION_BURN_IN,
                    evaluation_steps=CALIBRATION_STEPS,
                    initial_mode="random",
                    order="canonical",
                    step=steps,
                )
                _print_grid(measured, label=f"CALIBRATION step={steps}")
                score = -float(np.mean([row["objective_gap"] for row in measured]))
                reference_update = policy_retention.update_references(
                    actor,
                    {row["dual_lambda"]: row["objective"] for row in measured},
                )
                updated = ",".join(f"{dual:g}" for dual in reference_update.updated_duals)
                print(
                    f"[RETENTION] step={steps} updated_lambdas={updated or 'none'}",
                    flush=True,
                )
                updated_set = set(reference_update.updated_duals)
                for row, best_objective in zip(
                    measured,
                    reference_update.best_scores,
                    strict=True,
                ):
                    row["retention_reference_updated"] = row["dual_lambda"] in updated_set
                    row["retention_best_objective"] = best_objective
                manifest["policy_retention_best_objectives"] = {
                    str(dual): best
                    for dual, best in zip(
                        spec.dual_values,
                        reference_update.best_scores,
                        strict=True,
                    )
                }
                manifest["policy_retention_coefficients"] = list(
                    policy_retention.task_coefficients(spec.policy_retention_coefficient)
                    .detach()
                    .cpu()
                    .tolist()
                )
                calibration_rows.extend(measured)
                write_rows(root / "periodic_calibration.csv", calibration_rows)
                checkpoint = _checkpoint(
                    spec,
                    actor,
                    reward_critic,
                    cost_critic,
                    actor_optimizer,
                    reward_optimizer,
                    cost_optimizer,
                    reward_rates,
                    cost_rates,
                    dual_schedule,
                    restart_schedule,
                    vector,
                    permutation_rng,
                    reset_rng,
                    pcgrad_generator,
                    policy_retention,
                    steps,
                    updates,
                    score,
                )
                torch.save(checkpoint, root / "last.pt")
                if score > best_score:
                    best_score = score
                    best_step = steps
                    torch.save(checkpoint, root / "best.pt")
                manifest.update(
                    best_checkpoint_environment_steps=best_step,
                    best_calibration_score=best_score,
                    restart_schedule=asdict(restart_schedule),
                )
                write_json(root / "run_manifest.json", manifest)
                next_calibration = (
                    steps // settings.evaluation_interval + 1
                ) * settings.evaluation_interval

    best = torch.load(root / "best.pt", map_location=device, weights_only=False)
    actor.load_state_dict(best["actor"])
    final_rows = []
    evaluation_root = root / "evaluation"
    representative_trajectories = {}
    for initial_mode in ("fixed", "random"):
        for order in ("canonical", "random"):
            rows, trajectories = _evaluate_grid(
                actor,
                environment,
                spec,
                device,
                seeds=config.evaluation.seeds,
                burn_in_steps=config.evaluation.burn_in_steps,
                evaluation_steps=config.evaluation.evaluation_steps,
                initial_mode=initial_mode,
                order=order,
                step=best_step,
                save_root=evaluation_root,
            )
            final_rows.extend(rows)
            _print_grid(rows, label=f"FINAL {initial_mode}/{order}")
            if initial_mode == "random" and order == "canonical":
                representative_trajectories = trajectories
    write_rows(root / "summary.csv", final_rows)
    _plot_final(final_rows, image_root)
    preferred_trajectory_duals = (0.0, 2.5, 5.0, 7.0, 9.0, 10.0)
    trajectory_duals = (
        tuple(representative_trajectories)
        if len(representative_trajectories) <= len(preferred_trajectory_duals)
        else tuple(
            dual for dual in preferred_trajectory_duals if dual in representative_trajectories
        )
    )
    for dual in trajectory_duals:
        trajectory = representative_trajectories[dual]
        slug = str(int(dual)) if float(dual).is_integer() else str(dual).replace(".", "p")
        render_trajectory(
            environment,
            trajectory.position_history,
            image_root / f"trajectory_lambda{slug}.png",
            title=f"dual-conditioned | lambda={dual:g} | stochastic random initial",
        )
    manifest.update(
        status="completed",
        environment_steps=steps,
        updates=updates,
        duration_seconds=time.perf_counter() - started,
        best_checkpoint_environment_steps=best_step,
        best_calibration_score=best_score,
        formal_final_evaluation_completed=True,
    )
    write_json(root / "run_manifest.json", manifest)
    print(f"[DONE] best_step={best_step} output={root}", flush=True)


__all__ = ["train"]
