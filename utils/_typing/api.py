from typing import Literal

from pydantic import BaseModel as B


class OOD(B):
    method: Literal['ESTD', 'DIME']
    score: float
    
class Prematch(B):
    outcome: float
    OOD_method_value: list[OOD]