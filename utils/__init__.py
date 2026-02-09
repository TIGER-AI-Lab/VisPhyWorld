"""
共享的工具模块。

为了兼容依赖第三方模型（例如基于 DINO TorchHub 的指标模块），该文件会暴露
`trunc_normal_` 等常用函数，避免 `import utils` 时出现缺失。
"""

from __future__ import annotations

from typing import Any

try:
    from timm.models.layers import trunc_normal_ as _timm_trunc_normal_
except Exception:  # pragma: no cover - timm 不可用时回退
    _timm_trunc_normal_ = None  # type: ignore

if _timm_trunc_normal_ is not None:

    def trunc_normal_(tensor: Any, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> Any:
        """调用 timm 实现的截断正态初始化。"""
        return _timm_trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

else:  # 兜底使用 PyTorch 自带实现
    try:
        from torch.nn.init import trunc_normal_ as _torch_trunc_normal_
    except Exception as err:  # pragma: no cover - 极端环境才会触发

        def trunc_normal_(*_: Any, **__: Any) -> None:
            raise RuntimeError("未找到 trunc_normal_ 实现，无法初始化权重") from err

    else:

        def trunc_normal_(tensor: Any, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> Any:
            """使用 torch.nn.init.trunc_normal_ 的后备实现。"""
            return _torch_trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


__all__ = ["trunc_normal_"]
