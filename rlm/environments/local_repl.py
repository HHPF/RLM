import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import NonIsolatedEnv

# =============================================================================
# Safe Builtins
# =============================================================================

# Safe builtins - blocks dangerous operations like eval/exec/input
_SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


class LocalREPL(NonIsolatedEnv):
    """
    具有持久 Python 命名空间的本地 REPL 环境。
    在可以访问上下文数据的沙箱命名空间中执行代码。
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lm_handler_address = lm_handler_address
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self._lock = threading.Lock()

        # Setup globals, locals, and modules in environment.
        self.setup()

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """设置环境。"""
        # Create sandboxed globals
        self.globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: dict[str, Any] = {}

        # Track LLM calls made during code execution
        self._pending_llm_calls: list[RLMChatCompletion] = []

        # Add helper functions
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched

    def _final_var(self, variable_name: str) -> str:
        """返回变量的值作为最终答案。"""
        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            return str(self.locals[variable_name])
        return f"Error: Variable '{variable_name}' not found"

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        """通过与处理器的套接字连接查询语言模型。

        参数:
            prompt: 发送给语言模型的提示。
            model: 可选的模型名称（如果处理器有多个客户端）。
        """
        if not self.lm_handler_address:
            return "Error: No LM handler configured"

        try:
            request = LMRequest(prompt=prompt, model=model)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return f"Error: {response.error}"

            # Track this LLM call
            self._pending_llm_calls.append(
                response.chat_completion,
            )

            return response.chat_completion.response
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """同时用多个提示查询语言模型。

        参数:
            prompts: 发送给语言模型的提示列表。
            model: 可选的模型名称（如果处理器有多个客户端）。

        返回:
            与输入提示顺序相同的响应列表。
        """
        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(prompts)

        try:
            responses = send_lm_request_batched(self.lm_handler_address, prompts, model=model)

            results = []
            for response in responses:
                if not response.success:
                    results.append(f"Error: {response.error}")
                else:
                    # Track this LLM call in list of all calls -- we may want to do this hierarchically
                    self._pending_llm_calls.append(response.chat_completion)
                    results.append(response.chat_completion.response)

            return results
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)

    def load_context(self, context_payload: dict | list | str):
        """将上下文加载到环境中。"""
        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, "context.txt")
            with open(context_path, "w") as f:
                f.write(context_payload)
            self.execute_code(f"with open(r'{context_path}', 'r') as f:\n    context = f.read()")
        else:
            context_path = os.path.join(self.temp_dir, "context.json")
            with open(context_path, "w") as f:
                json.dump(context_payload, f)
            self.execute_code(
                f"import json\nwith open(r'{context_path}', 'r') as f:\n    context = json.load(f)"
            )

    @contextmanager
    def _capture_output(self):
        """线程安全的上下文管理器，用于捕获标准输出/错误。"""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self):
        """临时更改为执行的临时目录。"""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def execute_code(self, code: str) -> REPLResult:
        """在持久命名空间中执行代码并返回结果。"""
        start_time = time.perf_counter()

        # Clear pending LLM calls from previous execution
        self._pending_llm_calls = []

        with self._capture_output() as (stdout_buf, stderr_buf):
            with self._temp_cwd():
                try:
                    combined = {**self.globals, **self.locals}
                    exec(code, combined, combined)

                    # Update locals with new variables
                    for key, value in combined.items():
                        if key not in self.globals and not key.startswith("_"):
                            self.locals[key] = value

                    stdout = stdout_buf.getvalue()
                    stderr = stderr_buf.getvalue()
                except Exception as e:
                    stdout = stdout_buf.getvalue()
                    stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """清理临时目录并重置状态。"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        self.globals.clear()
        self.locals.clear()

    def __del__(self):
        self.cleanup()
