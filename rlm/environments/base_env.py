from abc import ABC, abstractmethod

from rlm.core.types import REPLResult


class BaseEnv(ABC):
    """
    RLM 用于交互的基本 REPL 类环境。主要类型是隔离和非隔离的，
    其中隔离环境位于与语言模型不同的机器上。
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


class IsolatedEnv(BaseEnv, ABC):
    """
    这些环境（例如 Prime Envs、Modal Envs）位于与语言模型完全分离的机器上，
    保证与语言模型进程的完全隔离。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


class NonIsolatedEnv(BaseEnv, ABC):
    """
    这些环境与语言模型运行在同一台机器上，并根据环境选择提供不同级别的隔离。
    最简单的默认选项是作为子进程运行的本地 Python REPL。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError
