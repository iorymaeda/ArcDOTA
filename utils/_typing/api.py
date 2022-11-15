from typing import Literal

from pydantic import BaseModel as B


class OOD(B):
    method: Literal['ESTD', 'DIME']
    score: float
    
class Prematch(B):
    outcome: float
    model_tags: list[str]
    response_time: float
    OOD_method_value: list[OOD]

class Error(B):
    status_code: int
    message: str
    error: str