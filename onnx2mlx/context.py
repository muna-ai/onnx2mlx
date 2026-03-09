#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from dataclasses import dataclass
from typing import Literal

Float64Mode = Literal["raise", "use_fp32"]

@dataclass(frozen=True)
class ConvertContext:
    """
    Runtime context threaded through every op handler.
    """
    float64_mode: Float64Mode = "raise"