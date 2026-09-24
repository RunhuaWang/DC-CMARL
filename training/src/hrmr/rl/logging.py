"""Phase 2 训练与 fixed-λ evaluation 的严格 CSV 日志。"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Any, TextIO

import numpy as np

OCCUPANCY_COLUMNS = (
    "occupancy_S1",
    "occupancy_S2",
    "occupancy_S3",
    "occupancy_S4",
    "occupancy_H1",
    "occupancy_H2",
    "occupancy_H3",
    "occupancy_H4",
)

TRAINING_LOG_COLUMNS = (
    "fixed_lambda",
    "training_seed",
    "device",
    "environment_steps",
    "agent_action_steps",
    "rollout_mean_reward",
    "rollout_mean_cost",
    "avg_reward_estimate",
    "avg_cost_estimate",
    "rollout_scalarized_objective",
    "actor_loss",
    "reward_critic_loss",
    "cost_critic_loss",
    "value_anchor_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "actor_grad_norm",
    "reward_critic_grad_norm",
    "cost_critic_grad_norm",
    "learning_rate",
    "entropy_coefficient",
    *OCCUPANCY_COLUMNS,
)

FORMAL_TRAINING_DIAGNOSTIC_COLUMNS = (
    "rollout_mean_num_distinct_targets",
    "rollout_all_hazardous_mode_rate",
    "rollout_stay_action_rate",
    "reward_differential_value_mean",
    "reward_differential_value_std",
    "reward_differential_value_abs_max",
    "first_all_hazardous_environment_step",
    "all_hazardous_mode_rate_since_first_hit",
    "all_hazardous_samples_since_first_hit",
    "all_hazardous_retained_next_transition_count",
    "all_hazardous_retention_opportunity_count",
    "all_hazardous_next_transition_retention_rate",
    "longest_consecutive_all_hazardous_vector_transitions",
    "first_instantaneous_optimal_mode_step",
    "instantaneous_optimal_mode_rate_since_first_hit",
    "instantaneous_optimal_mode_samples_since_first_hit",
    "instantaneous_optimal_mode_retained_next_transition_count",
    "instantaneous_optimal_mode_retention_opportunity_count",
    "instantaneous_optimal_mode_next_transition_retention_rate",
    "longest_consecutive_instantaneous_optimal_mode_vector_transitions",
    "actor_loss_agent_1",
    "actor_loss_agent_2",
    "actor_loss_agent_3",
    "actor_loss_agent_4",
    "actor_entropy_agent_1",
    "actor_entropy_agent_2",
    "actor_entropy_agent_3",
    "actor_entropy_agent_4",
    "unique_contribution_agent_1",
    "unique_contribution_agent_2",
    "unique_contribution_agent_3",
    "unique_contribution_agent_4",
    "duplicate_target_agent_1",
    "duplicate_target_agent_2",
    "duplicate_target_agent_3",
    "duplicate_target_agent_4",
    "non_target_agent_1",
    "non_target_agent_2",
    "non_target_agent_3",
    "non_target_agent_4",
)

EVALUATION_LOG_COLUMNS = (
    "record_type",
    "fixed_lambda",
    "training_seed",
    "evaluation_seed",
    "sampling_mode",
    "visualization_only",
    "evaluation_steps",
    "burn_in_steps",
    "whole_mean_reward",
    "whole_std_reward",
    "whole_mean_cost",
    "whole_std_cost",
    "whole_scalarized_objective",
    "steady_mean_reward",
    "steady_std_reward",
    "steady_mean_cost",
    "steady_std_cost",
    "steady_scalarized_objective",
    "whole_mean_num_distinct_targets",
    "steady_mean_num_distinct_targets",
    "whole_all_hazardous_mode_rate",
    "steady_all_hazardous_mode_rate",
    "whole_operating_mode_rate",
    "steady_operating_mode_rate",
    "whole_mean_hazardous_targets_covered",
    "steady_mean_hazardous_targets_covered",
    "whole_mean_safe_targets_covered",
    "steady_mean_safe_targets_covered",
    "whole_stay_action_rate",
    "steady_stay_action_rate",
    "whole_actor_entropy",
    "steady_actor_entropy",
    *OCCUPANCY_COLUMNS,
    "steady_local_cost_agent_1",
    "steady_local_cost_agent_2",
    "steady_local_cost_agent_3",
    "steady_local_cost_agent_4",
)


def _normalized_value(value: Any, *, column: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        raise TypeError(f"CSV column {column!r} must be scalar, not an ndarray")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"CSV column {column!r} must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return "" if value is None else value
    raise TypeError(f"CSV column {column!r} has unsupported type {type(value).__qualname__}")


class CSVLogger:
    """字段固定、即时 flush、支持安全 append 的通用 CSV writer。"""

    def __init__(
        self,
        path: str | Path,
        fieldnames: Sequence[str],
        *,
        required_fields: Iterable[str] | None = None,
        append: bool = False,
    ) -> None:
        self.path = Path(path)
        self.fieldnames = tuple(fieldnames)
        if not self.fieldnames or any(not field for field in self.fieldnames):
            raise ValueError("fieldnames must contain non-empty column names")
        if len(set(self.fieldnames)) != len(self.fieldnames):
            raise ValueError("fieldnames must not contain duplicates")
        self.required_fields = frozenset(
            self.fieldnames if required_fields is None else required_fields
        )
        unknown_required = self.required_fields - set(self.fieldnames)
        if unknown_required:
            raise ValueError(
                f"required_fields are absent from fieldnames: {sorted(unknown_required)}"
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = True
        if append and self.path.exists() and self.path.stat().st_size:
            with self.path.open("r", encoding="utf-8", newline="") as existing_file:
                existing_header = next(csv.reader(existing_file), None)
            if existing_header != list(self.fieldnames):
                raise ValueError("cannot append CSV row because the existing header does not match")
            write_header = False
        mode = "a" if append else "w"
        self._file: TextIO = self.path.open(mode, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.fieldnames,
            extrasaction="raise",
        )
        self._lock = Lock()
        self._closed = False
        self.rows_written = 0
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log(self, row: Mapping[str, Any]) -> None:
        """验证并写入一行；缺失规范字段或出现未声明字段时立即失败。"""

        if self._closed:
            raise RuntimeError("cannot write to a closed CSVLogger")
        if not isinstance(row, Mapping):
            raise TypeError("CSV row must be a mapping")
        missing = self.required_fields - set(row)
        if missing:
            raise ValueError(f"CSV row is missing required columns: {sorted(missing)}")
        extra = set(row) - set(self.fieldnames)
        if extra:
            raise ValueError(f"CSV row contains undeclared columns: {sorted(extra)}")
        normalized = {
            field: _normalized_value(row.get(field), column=field) for field in self.fieldnames
        }
        with self._lock:
            self._writer.writerow(normalized)
            self._file.flush()
            self.rows_written += 1

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> CSVLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class TrainingCSVLogger(CSVLogger):
    """覆盖 Phase 2 §16 全部必需训练字段的 logger。"""

    def __init__(
        self,
        path: str | Path,
        *,
        append: bool = False,
        extra_fields: Sequence[str] = (),
    ) -> None:
        fields = (*TRAINING_LOG_COLUMNS, *tuple(extra_fields))
        super().__init__(
            path,
            fields,
            required_fields=TRAINING_LOG_COLUMNS,
            append=append,
        )


class EvaluationCSVLogger(CSVLogger):
    """逐 seed 与 aggregate fixed-λ 结果的 logger。"""

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        super().__init__(path, EVALUATION_LOG_COLUMNS, append=append)


def _occupancy_values(target_ids: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    if len(target_ids) != len(values):
        raise ValueError("target ids and occupancy values must have equal length")
    mapping = dict(zip(target_ids, values, strict=True))
    expected_ids = tuple(column.removeprefix("occupancy_") for column in OCCUPANCY_COLUMNS)
    missing = [target_id for target_id in expected_ids if target_id not in mapping]
    if missing:
        raise ValueError(f"evaluation occupancy is missing targets: {missing}")
    return {f"occupancy_{target_id}": float(mapping[target_id]) for target_id in expected_ids}


def evaluation_rows(
    result: Any,
    *,
    training_seed: int,
) -> tuple[dict[str, Any], ...]:
    """把 ``FixedLambdaEvaluation`` 展平为逐 seed 加 aggregate CSV rows。"""

    rows = []
    sampling_mode = result.sampling_mode
    for trajectory in result.trajectories:
        local_costs = trajectory.steady_summary.mean_local_costs
        if local_costs is None or len(local_costs) != 4:
            raise ValueError("evaluation requires four steady agent-local cost means")
        rows.append(
            {
                "record_type": "seed",
                "fixed_lambda": result.fixed_lambda,
                "training_seed": training_seed,
                "evaluation_seed": trajectory.seed,
                "sampling_mode": sampling_mode,
                "visualization_only": result.visualization_only,
                "evaluation_steps": result.evaluation_steps,
                "burn_in_steps": result.burn_in_steps,
                "whole_mean_reward": trajectory.whole_reward,
                "whole_std_reward": 0.0,
                "whole_mean_cost": trajectory.whole_cost,
                "whole_std_cost": 0.0,
                "whole_scalarized_objective": trajectory.whole_scalarized_objective,
                "steady_mean_reward": trajectory.steady_reward,
                "steady_std_reward": 0.0,
                "steady_mean_cost": trajectory.steady_cost,
                "steady_std_cost": 0.0,
                "steady_scalarized_objective": trajectory.steady_scalarized_objective,
                "whole_mean_num_distinct_targets": (trajectory.whole_mean_num_distinct_targets),
                "steady_mean_num_distinct_targets": (trajectory.steady_mean_num_distinct_targets),
                "whole_all_hazardous_mode_rate": trajectory.whole_all_hazardous_mode_rate,
                "steady_all_hazardous_mode_rate": trajectory.steady_all_hazardous_mode_rate,
                "whole_operating_mode_rate": trajectory.whole_operating_mode_rate,
                "steady_operating_mode_rate": trajectory.steady_operating_mode_rate,
                "whole_mean_hazardous_targets_covered": (
                    trajectory.whole_mean_hazardous_targets_covered
                ),
                "steady_mean_hazardous_targets_covered": (
                    trajectory.steady_mean_hazardous_targets_covered
                ),
                "whole_mean_safe_targets_covered": (trajectory.whole_mean_safe_targets_covered),
                "steady_mean_safe_targets_covered": (trajectory.steady_mean_safe_targets_covered),
                "whole_stay_action_rate": trajectory.whole_stay_action_rate,
                "steady_stay_action_rate": trajectory.steady_stay_action_rate,
                "whole_actor_entropy": trajectory.whole_actor_entropy,
                "steady_actor_entropy": trajectory.steady_actor_entropy,
                **_occupancy_values(
                    result.target_ids,
                    trajectory.steady_target_occupancy,
                ),
                **{
                    f"steady_local_cost_agent_{index + 1}": float(value)
                    for index, value in enumerate(local_costs)
                },
            }
        )

    aggregate = result.aggregate
    rows.append(
        {
            "record_type": "aggregate",
            "fixed_lambda": result.fixed_lambda,
            "training_seed": training_seed,
            "evaluation_seed": "",
            "sampling_mode": sampling_mode,
            "visualization_only": result.visualization_only,
            "evaluation_steps": result.evaluation_steps,
            "burn_in_steps": result.burn_in_steps,
            "whole_mean_reward": aggregate.whole_mean_reward,
            "whole_std_reward": aggregate.whole_std_reward,
            "whole_mean_cost": aggregate.whole_mean_cost,
            "whole_std_cost": aggregate.whole_std_cost,
            "whole_scalarized_objective": aggregate.whole_mean_scalarized_objective,
            "steady_mean_reward": aggregate.steady_mean_reward,
            "steady_std_reward": aggregate.steady_std_reward,
            "steady_mean_cost": aggregate.steady_mean_cost,
            "steady_std_cost": aggregate.steady_std_cost,
            "steady_scalarized_objective": aggregate.steady_mean_scalarized_objective,
            "whole_mean_num_distinct_targets": (aggregate.whole_mean_num_distinct_targets),
            "steady_mean_num_distinct_targets": (aggregate.steady_mean_num_distinct_targets),
            "whole_all_hazardous_mode_rate": (aggregate.whole_mean_all_hazardous_mode_rate),
            "steady_all_hazardous_mode_rate": (aggregate.steady_mean_all_hazardous_mode_rate),
            "whole_operating_mode_rate": aggregate.whole_mean_operating_mode_rate,
            "steady_operating_mode_rate": aggregate.steady_mean_operating_mode_rate,
            "whole_mean_hazardous_targets_covered": (
                aggregate.whole_mean_hazardous_targets_covered
            ),
            "steady_mean_hazardous_targets_covered": (
                aggregate.steady_mean_hazardous_targets_covered
            ),
            "whole_mean_safe_targets_covered": aggregate.whole_mean_safe_targets_covered,
            "steady_mean_safe_targets_covered": aggregate.steady_mean_safe_targets_covered,
            "whole_stay_action_rate": aggregate.whole_mean_stay_action_rate,
            "steady_stay_action_rate": aggregate.steady_mean_stay_action_rate,
            "whole_actor_entropy": aggregate.whole_mean_actor_entropy,
            "steady_actor_entropy": aggregate.steady_mean_actor_entropy,
            **_occupancy_values(
                result.target_ids,
                aggregate.steady_mean_target_occupancy,
            ),
            **{
                f"steady_local_cost_agent_{index + 1}": float(value)
                for index, value in enumerate(aggregate.steady_mean_local_costs)
            },
        }
    )
    return tuple(rows)


def write_evaluation_csv(
    path: str | Path,
    result: Any,
    *,
    training_seed: int,
    append: bool = False,
) -> Path:
    """将完整 per-seed/aggregate evaluation 一次写入 CSV。"""

    with EvaluationCSVLogger(path, append=append) as logger:
        for row in evaluation_rows(result, training_seed=training_seed):
            logger.log(row)
    return Path(path)


__all__ = [
    "EVALUATION_LOG_COLUMNS",
    "FORMAL_TRAINING_DIAGNOSTIC_COLUMNS",
    "OCCUPANCY_COLUMNS",
    "TRAINING_LOG_COLUMNS",
    "CSVLogger",
    "EvaluationCSVLogger",
    "TrainingCSVLogger",
    "evaluation_rows",
    "write_evaluation_csv",
]
