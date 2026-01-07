from abc import ABC, abstractmethod
from typing import Any

from rlm.core.types import UsageSummary


class BaseLM(ABC):
    """
    所有语言模型路由器/客户端的基类。当 RLM 进行子调用时，它目前
    以模型无关的方式进行，因此此类为所有语言模型提供了基础接口。
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def completion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary:
        """获取所有模型调用的成本摘要。"""
        raise NotImplementedError

    @abstractmethod
    def get_last_usage(self) -> UsageSummary:
        """获取模型的最后一次成本摘要。"""
        raise NotImplementedError
