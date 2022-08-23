from typing import Literal

from pydantic import BaseModel as B


class Prematch(B):
    outcome: float
    OOD_value: float | None
    OOD_method: Literal['ensemble_std'] | None