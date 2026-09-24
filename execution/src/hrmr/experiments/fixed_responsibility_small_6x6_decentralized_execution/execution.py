"""固定 dual-conditioned actors 的无中心执行与结果记录。"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hrmr.constants import Action
from hrmr.experiments.assigned_targets_ppo.environment import DEFAULT_ASSIGNMENTS
from hrmr.experiments.fixed_responsibility_small_6x6_dual_conditioned_mappo.conditioning import (
    DUAL_ACTOR_INPUT_DIM,
)
from hrmr.experiments.fixed_responsibility_small_6x6_dual_conditioned_mappo.config import (
    load_config as load_training_config,
)
from hrmr.experiments.fixed_responsibility_small_6x6_dual_conditioned_mappo.config import (
    load_environment as load_training_environment,
)
from hrmr.experiments.fixed_responsibility_small_6x6_dual_conditioned_mappo.networks import (
    FiLMIndependentActors,
)
from hrmr.experiments.fixed_responsibility_small_6x6_dual_conditioned_mappo.theory import (
    MODE_ALLOCATIONS,
)
from hrmr.experiments.fixed_responsibility_small_6x6_mappo.environment import (
    SmallSixBySixRandomEvaluationEnvironment,
)
from hrmr.rendering import render_layout, render_trajectory
from hrmr.rl.device import resolve_device

from .config import DecentralizedExecutionConfig, load_config
from .dual_update import DecentralizedDualController

POLICY_RNG_DOMAIN = 0x44454350
CHECKPOINT_FORMAT = "fixed_responsibility_small_6x6_dual_conditioned_mappo_v1"


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_existing(data_root: Path, image_root: Path, profile: str):
    paths = (data_root, image_root)
    if any(path.exists() and any(path.iterdir()) for path in paths):
        manifest_path = data_root / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"refusing to overwrite unrecognized output: {data_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("profile") != profile:
            raise FileExistsError("existing output belongs to another experiment")
        if manifest.get("status") == "running":
            raise FileExistsError("existing output is marked running")
    staged = []
    for path in paths:
        if not path.exists() or not any(path.iterdir()):
            continue
        backup = path.with_name(f".{path.name}.overwrite-backup")
        if backup.exists():
            raise FileExistsError(f"stale overwrite backup exists: {backup}")
        path.replace(backup)
        staged.append((path, backup))
    return staged


def _restore_staged(staged) -> None:
    for original, backup in reversed(staged):
        if original.exists():
            shutil.rmtree(original)
        backup.replace(original)


def _discard_staged(staged) -> None:
    for _, backup in staged:
        shutil.rmtree(backup)


def _load_frozen_actor(
    settings: DecentralizedExecutionConfig,
    device: torch.device,
) -> tuple[FiLMIndependentActors, dict]:
    checkpoint = torch.load(settings.checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not the retained dual-conditioned MAPPO format")
    if checkpoint.get("conditioning_max") != settings.dual_max:
        raise ValueError("checkpoint conditioning range does not match execution range")
    if checkpoint.get("actor_conditioning") != "continuous_film":
        raise ValueError("checkpoint does not contain the retained FiLM actor structure")
    seeds = tuple(checkpoint.get("actor_initialization_seeds", ()))
    if len(seeds) != 4:
        raise ValueError("checkpoint is missing four actor initialization seeds")
    actor = FiLMIndependentActors(
        num_agents=4,
        input_dim=DUAL_ACTOR_INPUT_DIM,
        hidden_dims=(128, 128),
        num_actions=5,
        activation="tanh",
        initialization_seeds=seeds,
    ).to(device)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()
    actor.requires_grad_(False)
    return actor, checkpoint


def _condition_local_duals(
    observations: np.ndarray,
    local_duals: np.ndarray,
    dual_max: float,
) -> np.ndarray:
    """为每个 agent 追加自己的 λ_i/Λ，而不是读取集中式共享变量。"""

    raw = np.asarray(observations, dtype=np.float64)
    duals = np.asarray(local_duals, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (4, 56):
        raise ValueError("observations must have shape [num_runs,4,56]")
    if duals.shape != raw.shape[:2]:
        raise ValueError("local_duals must have shape [num_runs,4]")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(duals)):
        raise ValueError("actor inputs must be finite")
    if np.any((duals < 0.0) | (duals > dual_max)):
        raise ValueError("local dual copies must lie in the trained interval")
    return np.concatenate((raw, (duals / dual_max)[..., None]), axis=-1)


def _sample_actions(
    actor: FiLMIndependentActors,
    policy_inputs: np.ndarray,
    generators: tuple[np.random.Generator, ...],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(policy_inputs, dtype=torch.float32, device=device)
    probabilities = actor.distribution(tensor).probs.detach().cpu().numpy()
    uniforms = np.stack([generator.random(4) for generator in generators])
    cumulative = np.cumsum(probabilities, axis=-1)
    cumulative[..., -1] = 1.0
    actions = np.sum(uniforms[..., None] > cumulative, axis=-1).astype(np.int64)
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0.0)
    entropies = -np.sum(probabilities * log_probabilities, axis=-1)
    return actions, entropies


def _mode_indicators(occupancies: np.ndarray, target_ids: tuple[str, ...]) -> np.ndarray:
    result = np.zeros(len(MODE_ALLOCATIONS), dtype=np.float64)
    if int(np.sum(occupancies)) != 4:
        return result
    for index, allocation in enumerate(MODE_ALLOCATIONS.values()):
        required = tuple(target_ids.index(target) for target in allocation)
        result[index] = float(np.all(occupancies[list(required)] == 1))
    return result


def _threshold_slug(threshold: float) -> str:
    return (
        str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
    )


def _aggregate_threshold_rows(seed_rows: list[dict], thresholds: tuple[float, ...]) -> list[dict]:
    result = []
    metrics = (
        "mean_reward",
        "mean_global_cost",
        "constraint_violation",
        "tail_mean_reward",
        "tail_mean_global_cost",
        "tail_constraint_violation",
        "tail_mean_dual",
        "final_mean_dual",
        "max_dual_disagreement",
    )
    for threshold in thresholds:
        selected = [row for row in seed_rows if row["constraint_threshold"] == threshold]
        row = {"constraint_threshold": threshold, "num_seeds": len(selected)}
        for metric in metrics:
            values = np.asarray([item[metric] for item in selected], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        for mode in MODE_ALLOCATIONS:
            values = np.asarray([item[f"tail_mode_{mode}_rate"] for item in selected])
            row[f"tail_mode_{mode}_rate_mean"] = float(np.mean(values))
        result.append(row)
    return result


def _plot_dynamics(block_rows: list[dict], settings: DecentralizedExecutionConfig) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for threshold in settings.constraint_thresholds:
        rows = [
            row
            for row in block_rows
            if row["constraint_threshold"] == threshold
            and row["seed"] == settings.representative_seed
        ]
        steps = [row["environment_steps"] for row in rows]
        axes[0].plot(steps, [row["dual_after_mean"] for row in rows], label=f"c={threshold:g}")
        axes[1].plot(
            steps,
            [row["window_global_cost"] for row in rows],
            label=f"c={threshold:g}",
        )
        axes[1].axhline(threshold, color="black", linewidth=0.5, alpha=0.15)
    axes[0].set_ylabel("Mean local dual copy")
    axes[0].set_ylim(settings.dual_min - 0.2, settings.dual_max + 0.2)
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Window global cost")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(settings.image_directory / "dual_and_cost_dynamics.png", dpi=180)
    plt.close(figure)


def _plot_threshold_response(threshold_rows: list[dict], settings) -> None:
    thresholds = np.asarray([row["constraint_threshold"] for row in threshold_rows])
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].errorbar(
        thresholds,
        [row["tail_mean_reward_mean"] for row in threshold_rows],
        yerr=[row["tail_mean_reward_std"] for row in threshold_rows],
        marker="o",
    )
    axes[0].set_ylabel("Tail mean reward")
    axes[1].errorbar(
        thresholds,
        [row["tail_mean_global_cost_mean"] for row in threshold_rows],
        yerr=[row["tail_mean_global_cost_std"] for row in threshold_rows],
        marker="o",
        color="tab:red",
        label="observed cost",
    )
    axes[1].plot(thresholds, thresholds, linestyle="--", color="black", label="threshold c")
    axes[1].set(xlabel="Constraint threshold c", ylabel="Tail mean global cost")
    for axis in axes:
        axis.grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(settings.image_directory / "threshold_response.png", dpi=180)
    plt.close(figure)


def execute(
    config_path: str | Path,
    *,
    device_override: str | None = None,
    overwrite_existing: bool = True,
) -> Path:
    """加载 best checkpoint，并按确认设置执行全部阈值和随机种子。"""

    settings = load_config(config_path)
    device = resolve_device(device_override or settings.device)
    staged = (
        _stage_existing(settings.data_directory, settings.image_directory, settings.profile)
        if overwrite_existing
        else []
    )
    if not overwrite_existing and (
        settings.data_directory.exists() or settings.image_directory.exists()
    ):
        raise FileExistsError("decentralized execution output already exists")
    manifest = {
        "profile": settings.profile,
        "status": "running",
        "settings": {
            **asdict(settings),
            "source_path": str(settings.source_path),
            "checkpoint_path": str(settings.checkpoint_path),
            "training_config_path": str(settings.training_config_path),
            "data_directory": str(settings.data_directory),
            "image_directory": str(settings.image_directory),
        },
        "checkpoint_sha256": _sha256(settings.checkpoint_path),
    }
    # JSON 不支持 tuple 嵌套路径等对象。
    manifest["settings"]["graph"] = [list(neighbors) for neighbors in settings.graph]
    try:
        settings.data_directory.mkdir(parents=True, exist_ok=True)
        settings.image_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(settings.source_path, settings.data_directory / "config.toml")
        shutil.copy2(
            settings.training_config_path,
            settings.data_directory / "training_config.toml",
        )
        _write_json(settings.data_directory / "run_manifest.json", manifest)
        _run(settings, device, manifest)
    except KeyboardInterrupt:
        manifest.update(status="interrupted")
        if settings.data_directory.exists():
            _write_json(settings.data_directory / "run_manifest.json", manifest)
        _restore_staged(staged)
        raise
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        if settings.data_directory.exists():
            _write_json(settings.data_directory / "run_manifest.json", manifest)
        _restore_staged(staged)
        raise
    _discard_staged(staged)
    return settings.data_directory


def _run(settings: DecentralizedExecutionConfig, device: torch.device, manifest: dict) -> None:
    training_spec = load_training_config(settings.training_config_path)
    environment_config = load_training_environment(training_spec)
    actor, checkpoint = _load_frozen_actor(settings, device)
    parameter_snapshot = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    thresholds = settings.constraint_thresholds
    seeds = settings.evaluation_seeds
    run_keys = tuple((threshold, seed) for threshold in thresholds for seed in seeds)
    environments = tuple(
        SmallSixBySixRandomEvaluationEnvironment(
            environment_config,
            assignments=DEFAULT_ASSIGNMENTS,
        )
        for _ in run_keys
    )
    reset_results = tuple(
        environment.reset(seed=seed)
        for environment, (_, seed) in zip(environments, run_keys, strict=True)
    )
    observations = np.stack([result[0] for result in reset_results])
    initial_positions = np.stack([result[2]["agent_positions"] for result in reset_results])
    controllers = tuple(
        DecentralizedDualController(
            graph=settings.graph,
            communication_rounds=settings.communication_rounds,
            initial_dual=settings.initial_dual,
            constraint_threshold=threshold,
            step_size=settings.dual_step_size,
            dual_min=settings.dual_min,
            dual_max=settings.dual_max,
            horizon=settings.cost_estimation_horizon,
        )
        for threshold, _ in run_keys
    )
    # 相同 seed 在不同阈值下使用相同 uniform random stream，形成 common random numbers。
    generators = tuple(
        np.random.default_rng(np.random.SeedSequence((seed, POLICY_RNG_DOMAIN)))
        for _, seed in run_keys
    )
    num_runs = len(run_keys)
    total_reward = np.zeros(num_runs)
    total_cost = np.zeros(num_runs)
    total_local_cost = np.zeros((num_runs, 4))
    tail_reward = np.zeros(num_runs)
    tail_cost = np.zeros(num_runs)
    target_counts = np.zeros((num_runs, 8))
    tail_target_counts = np.zeros((num_runs, 8))
    mode_counts = np.zeros((num_runs, len(MODE_ALLOCATIONS)))
    tail_mode_counts = np.zeros_like(mode_counts)
    stay_counts = np.zeros(num_runs)
    tail_stay_counts = np.zeros(num_runs)
    entropy_sums = np.zeros((num_runs, 4))
    tail_entropy_sums = np.zeros((num_runs, 4))
    tail_dual_sums = np.zeros((num_runs, 4))
    max_disagreement = np.zeros(num_runs)
    window_rewards = np.zeros(num_runs)
    window_costs = np.zeros(num_runs)
    block_rows: list[dict] = []
    representative_indices = {
        threshold: run_keys.index((threshold, settings.representative_seed))
        for threshold in thresholds
    }
    representative = {
        threshold: {
            "positions": [initial_positions[index].astype(np.int8)],
            "rewards": [],
            "global_costs": [],
            "local_duals": [],
            "actions": [],
            "occupancies": [],
        }
        for threshold, index in representative_indices.items()
    }
    render_layout(
        environment_config,
        settings.image_directory / "layout.png",
        title="6x6 decentralized dual execution",
    )
    started = time.perf_counter()
    tail_start = settings.total_environment_steps // 2
    target_ids = tuple(environment_config.target_ids)

    with torch.inference_mode():
        for step in range(settings.total_environment_steps):
            local_duals = np.stack([controller.dual_values for controller in controllers])
            policy_inputs = _condition_local_duals(observations, local_duals, settings.dual_max)
            actions, entropies = _sample_actions(actor, policy_inputs, generators, device)
            next_observations = []
            block_finished = False
            for run_index, (environment, joint_actions) in enumerate(
                zip(environments, actions, strict=True)
            ):
                result = environment.step(joint_actions)
                next_obs, _, reward, local_costs, terminated, truncated, info = result
                if terminated or truncated:
                    raise AssertionError("decentralized execution must remain continuing")
                local_costs = np.asarray(local_costs, dtype=np.float64)
                global_cost = float(info["global_cost"])
                if not np.isclose(global_cost, np.sum(local_costs), rtol=0.0, atol=1e-12):
                    raise AssertionError("global cost differs from the sum of local costs")
                occupancy = np.asarray(info["target_coverage"], dtype=np.int8)
                mode = _mode_indicators(occupancy, target_ids)
                total_reward[run_index] += reward
                total_cost[run_index] += global_cost
                total_local_cost[run_index] += local_costs
                target_counts[run_index] += occupancy
                mode_counts[run_index] += mode
                stay_counts[run_index] += np.mean(joint_actions == int(Action.STAY))
                entropy_sums[run_index] += entropies[run_index]
                window_rewards[run_index] += reward
                window_costs[run_index] += global_cost
                if step >= tail_start:
                    tail_reward[run_index] += reward
                    tail_cost[run_index] += global_cost
                    tail_target_counts[run_index] += occupancy
                    tail_mode_counts[run_index] += mode
                    tail_stay_counts[run_index] += np.mean(joint_actions == int(Action.STAY))
                    tail_entropy_sums[run_index] += entropies[run_index]
                    tail_dual_sums[run_index] += local_duals[run_index]
                update = controllers[run_index].observe(local_costs)
                if update is not None:
                    block_finished = True
                    disagreement = float(np.ptp(update.dual_after))
                    max_disagreement[run_index] = max(max_disagreement[run_index], disagreement)
                    threshold, seed = run_keys[run_index]
                    row = {
                        "dual_update": (step + 1) // settings.cost_estimation_horizon,
                        "environment_steps": step + 1,
                        "constraint_threshold": threshold,
                        "seed": seed,
                        "window_mean_reward": window_rewards[run_index]
                        / settings.cost_estimation_horizon,
                        "window_global_cost": window_costs[run_index]
                        / settings.cost_estimation_horizon,
                        "window_constraint_violation": max(
                            window_costs[run_index] / settings.cost_estimation_horizon - threshold,
                            0.0,
                        ),
                        "dual_before_mean": float(np.mean(update.dual_before)),
                        "dual_after_mean": float(np.mean(update.dual_after)),
                        "dual_disagreement": disagreement,
                    }
                    for agent in range(4):
                        row[f"local_cost_estimate_A{agent + 1}"] = float(
                            update.local_cost_estimates[agent]
                        )
                        row[f"global_cost_estimate_A{agent + 1}"] = float(
                            update.global_cost_estimates[agent]
                        )
                        row[f"dual_before_A{agent + 1}"] = float(update.dual_before[agent])
                        row[f"dual_after_A{agent + 1}"] = float(update.dual_after[agent])
                    block_rows.append(row)
                    window_rewards[run_index] = 0.0
                    window_costs[run_index] = 0.0
                next_observations.append(next_obs)
                threshold, seed = run_keys[run_index]
                if seed == settings.representative_seed:
                    record = representative[threshold]
                    record["positions"].append(np.asarray(info["agent_positions"], dtype=np.int8))
                    record["rewards"].append(float(reward))
                    record["global_costs"].append(global_cost)
                    record["local_duals"].append(local_duals[run_index].astype(np.float32))
                    record["actions"].append(np.asarray(joint_actions, dtype=np.int8))
                    record["occupancies"].append(occupancy)
            observations = np.stack(next_observations)

            if block_finished:
                update_index = (step + 1) // settings.cost_estimation_horizon
                if update_index == 1 or update_index % settings.progress_every_dual_updates == 0:
                    summaries = []
                    for threshold in thresholds:
                        indices = [
                            index for index, (value, _) in enumerate(run_keys) if value == threshold
                        ]
                        mean_dual = np.mean(
                            [np.mean(controllers[index].dual_values) for index in indices]
                        )
                        recent = block_rows[-num_runs:]
                        mean_cost = np.mean(
                            [
                                row["window_global_cost"]
                                for row in recent
                                if row["constraint_threshold"] == threshold
                            ]
                        )
                        summaries.append(
                            f"c={threshold:g}:lambda={mean_dual:.2f},C={mean_cost:.3f}"
                        )
                    print(
                        f"[EXEC] update={update_index}/{settings.num_dual_updates} "
                        f"step={step + 1}/{settings.total_environment_steps} "
                        f"elapsed={time.perf_counter() - started:.0f}s | " + " | ".join(summaries),
                        flush=True,
                    )

    tail_steps = settings.total_environment_steps - tail_start
    seed_rows = []
    for run_index, (threshold, seed) in enumerate(run_keys):
        final_duals = controllers[run_index].dual_values
        row = {
            "constraint_threshold": threshold,
            "seed": seed,
            "mean_reward": total_reward[run_index] / settings.total_environment_steps,
            "mean_global_cost": total_cost[run_index] / settings.total_environment_steps,
            "constraint_violation": max(
                total_cost[run_index] / settings.total_environment_steps - threshold,
                0.0,
            ),
            "tail_mean_reward": tail_reward[run_index] / tail_steps,
            "tail_mean_global_cost": tail_cost[run_index] / tail_steps,
            "tail_constraint_violation": max(
                tail_cost[run_index] / tail_steps - threshold,
                0.0,
            ),
            "tail_mean_dual": float(np.mean(tail_dual_sums[run_index] / tail_steps)),
            "final_mean_dual": float(np.mean(final_duals)),
            "max_dual_disagreement": max_disagreement[run_index],
            "stay_action_rate": stay_counts[run_index] / settings.total_environment_steps,
            "tail_stay_action_rate": tail_stay_counts[run_index] / tail_steps,
        }
        for agent in range(4):
            row[f"mean_local_cost_A{agent + 1}"] = (
                total_local_cost[run_index, agent] / settings.total_environment_steps
            )
            row[f"mean_entropy_A{agent + 1}"] = (
                entropy_sums[run_index, agent] / settings.total_environment_steps
            )
            row[f"tail_mean_entropy_A{agent + 1}"] = (
                tail_entropy_sums[run_index, agent] / tail_steps
            )
            row[f"final_dual_A{agent + 1}"] = float(final_duals[agent])
        for target_index, target_id in enumerate(target_ids):
            row[f"occupancy_{target_id}"] = (
                target_counts[run_index, target_index] / settings.total_environment_steps
            )
            row[f"tail_occupancy_{target_id}"] = (
                tail_target_counts[run_index, target_index] / tail_steps
            )
        for mode_index, mode in enumerate(MODE_ALLOCATIONS):
            row[f"mode_{mode}_rate"] = (
                mode_counts[run_index, mode_index] / settings.total_environment_steps
            )
            row[f"tail_mode_{mode}_rate"] = tail_mode_counts[run_index, mode_index] / tail_steps
        seed_rows.append(row)

    threshold_rows = _aggregate_threshold_rows(seed_rows, thresholds)
    _write_rows(settings.data_directory / "seed_level_results.csv", seed_rows)
    _write_rows(settings.data_directory / "dual_update_log.csv", block_rows)
    _write_rows(settings.data_directory / "threshold_summary.csv", threshold_rows)
    for threshold, record in representative.items():
        slug = _threshold_slug(threshold)
        positions = np.asarray(record["positions"], dtype=np.int8)
        np.savez_compressed(
            settings.data_directory / f"trajectory_c{slug}_seed{settings.representative_seed}.npz",
            positions=positions,
            rewards=np.asarray(record["rewards"], dtype=np.float32),
            global_costs=np.asarray(record["global_costs"], dtype=np.float32),
            local_duals=np.asarray(record["local_duals"], dtype=np.float32),
            actions=np.asarray(record["actions"], dtype=np.int8),
            occupancies=np.asarray(record["occupancies"], dtype=np.int8),
        )
        render_trajectory(
            environment_config,
            positions,
            settings.image_directory / f"trajectory_c{slug}_seed{settings.representative_seed}.png",
            title=f"Decentralized execution | c={threshold:g} | lambda0=0",
        )
    _plot_dynamics(block_rows, settings)
    _plot_threshold_response(threshold_rows, settings)

    if any(
        not torch.equal(value.detach().cpu(), parameter_snapshot[name])
        for name, value in actor.state_dict().items()
    ):
        raise AssertionError("decentralized execution modified frozen actor parameters")
    manifest.update(
        status="completed",
        duration_seconds=time.perf_counter() - started,
        checkpoint_environment_steps=checkpoint.get("environment_steps"),
        checkpoint_calibration_score=checkpoint.get("calibration_score"),
        total_trajectories=len(run_keys),
        total_environment_interactions=len(run_keys) * settings.total_environment_steps,
        max_dual_disagreement=float(np.max(max_disagreement)),
    )
    _write_json(settings.data_directory / "run_manifest.json", manifest)
    print(f"[DONE] output={settings.data_directory}", flush=True)


__all__ = ["execute"]
