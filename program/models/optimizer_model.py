from pydantic import BaseModel
from pydantic.config import ConfigDict

class CustomBaseModel(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)

class AdamConfig(CustomBaseModel):
    type: str = "adam"
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0

class SGDConfig(CustomBaseModel):
    type: str = "sgd"
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0