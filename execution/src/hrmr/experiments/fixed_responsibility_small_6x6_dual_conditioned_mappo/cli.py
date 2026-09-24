"""6×6 fixed-responsibility dual-conditioned MAPPO 命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import (
    CONFIG_PATH,
    default_image_output,
    default_output,
    load_config,
    load_environment,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    default_config: Path = CONFIG_PATH,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda", "mps"), default="cpu")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="若同一 seed0 结果已存在则拒绝启动；默认安全覆盖同身份结果",
    )
    arguments = parser.parse_args(argv)
    try:
        spec = load_config(arguments.config)
        environment = load_environment(spec)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    output = (
        default_output(spec)
        if arguments.output_root is None
        else arguments.output_root.expanduser().resolve()
    )
    image_output = (
        default_image_output(spec) if arguments.output_root is None else output / "images"
    )
    if arguments.check_only:
        settings = spec.base.training
        print(f"Configuration OK: {spec.base.experiment.profile}")
        print(f"Geometry: {environment.grid_width}x{environment.grid_height}; owner flag enabled")
        print(f"Dual grid: {spec.dual_values}")
        print(
            f"Balanced batch: {settings.num_parallel_envs} envs = "
            f"{len(spec.dual_values)} lambdas x {spec.environments_per_dual} envs/lambda"
        )
        print(f"Lambda-to-environment assignment: {spec.dual_assignment}")
        print(f"Actor conditioning: {spec.actor_conditioning}")
        print(f"Critic conditioning: {spec.critic_conditioning}")
        print(f"Actor gradient aggregation: {spec.actor_gradient_aggregation}")
        print(
            f"Policy retention: {spec.policy_retention}; "
            f"coefficient={spec.policy_retention_coefficient:g}; "
            f"anchors/lambda={spec.policy_retention_anchor_capacity}"
        )
        print(
            "Adaptive retention: "
            f"target_kl={spec.policy_retention_target_kl:g}; "
            f"rate={spec.policy_retention_adaptation_rate:g}; "
            f"coefficient_range=[{spec.policy_retention_min_coefficient:g}, "
            f"{spec.policy_retention_max_coefficient:g}]; "
            f"every_minibatch={spec.policy_retention_apply_every_minibatch}"
        )
        print(f"Terminal progress interval: {spec.progress_every_updates} updates")
        print(
            f"rollout={settings.rollout_length} batch="
            f"{settings.num_parallel_envs * settings.rollout_length} "
            f"minibatch={settings.minibatch_size} n={settings.n_step} "
            f"total_steps={settings.total_environment_steps}"
        )
        print("Independent 57D actors; centralized 17D reward/cost critics")
        print("Per-lambda rho and advantage normalization; lambda-stratified PPO minibatches")
        print(
            "Formal final evaluation: stochastic, seeds 1000..1019, "
            f"{len(spec.dual_values)} configured lambdas"
        )
        print(f"Data output: {output}")
        print(f"Image output: {image_output}")
        return 0

    from .trainer import train

    try:
        train(
            arguments.config,
            arguments.output_root,
            device_override=arguments.device,
            overwrite_existing=not arguments.no_overwrite,
        )
    except KeyboardInterrupt:
        print("[STOPPED] Training interrupted; previous complete output restored.", flush=True)
        return 130
    except FileExistsError as error:
        parser.error(str(error))
    return 0


__all__ = ["main"]
