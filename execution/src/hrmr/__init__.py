"""HRMR Phase 1 基础静态多机器人监控环境。"""

from hrmr.config import HRMRConfig, load_config
from hrmr.constants import Action

__all__ = ["Action", "HRMRConfig", "HRMREnvironment", "load_config"]
__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    """延迟导入环境类，保持纯函数子模块可独立使用。"""

    if name == "HRMREnvironment":
        from hrmr.environment import HRMREnvironment

        return HRMREnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
