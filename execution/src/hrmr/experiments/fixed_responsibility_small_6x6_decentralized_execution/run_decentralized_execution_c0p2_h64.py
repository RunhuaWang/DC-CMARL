"""IDE 直接运行：c=0.2、H=64、1024 次 dual 更新。"""

from hrmr.experiments import (
    fixed_responsibility_small_6x6_decentralized_execution as experiment,
)

if __name__ == "__main__":
    raise SystemExit(experiment.run_single_threshold(experiment.C0P2_H64_CONFIG_PATH))
