from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseExecutor(ABC):
  @abstractmethod
  def execute(self) -> Any:
    """Execute this executor's duty and return a typed result."""
    pass
