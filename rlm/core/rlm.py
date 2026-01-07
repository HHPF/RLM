import time
from contextlib import contextmanager
from typing import Any

from rlm.clients import BaseLM, get_client
from rlm.core.lm_handler import LMHandler
from rlm.core.types import (
    ClientBackend,
    CodeBlock,
    EnvironmentType,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
)
from rlm.environments import BaseEnv, get_environment
from rlm.logger import RLMLogger, VerbosePrinter
from rlm.utils.parsing import (
    find_code_blocks,
    find_final_answer,
    format_iteration,
)
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from rlm.utils.rlm_utils import filter_sensitive_keys


class RLM:
    """
    递归语言模型类，用户实例化并在其任务上运行。

    每个 completion() 调用都会生成自己的环境和语言模型处理器，
    当调用完成时会被清理。
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        custom_system_prompt: str | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
    ):
        """
        参数:
            backend: RLM 使用的后端模型。
            backend_kwargs: 传递给后端的关键字参数。
            environment: RLM 使用的执行环境。
            environment_kwargs: 传递给环境的关键字参数。
            depth: RLM 的当前深度（从0开始）。
            max_depth: RLM 的最大深度。目前仅支持深度1。
            max_iterations: RLM 的最大迭代次数。
            custom_system_prompt: 用于 RLM 的自定义系统提示。
            other_backends: 环境可以用来进行子调用的其他客户端后端列表。
            other_backend_kwargs: 传递给其他客户端后端的关键字参数（顺序与 other_backends 匹配）。
            logger: 用于 RLM 的日志记录器。
            verbose: 是否在控制台以富文本形式打印详细输出。
        """
        # Store config for spawning per-completion
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.environment_type = environment
        self.environment_kwargs = (
            environment_kwargs.copy() if environment_kwargs is not None else {}
        )
        self.other_backends = other_backends
        self.other_backend_kwargs = other_backend_kwargs

        self.depth = depth
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.system_prompt = custom_system_prompt if custom_system_prompt else RLM_SYSTEM_PROMPT
        self.logger = logger
        self.verbose = VerbosePrinter(enabled=verbose)

        # Log metadata if logger is provided
        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=backend_kwargs.get("model_name", "unknown")
                if backend_kwargs
                else "unknown",
                max_depth=max_depth,
                max_iterations=max_iterations,
                backend=backend,
                backend_kwargs=filter_sensitive_keys(backend_kwargs) if backend_kwargs else {},
                environment_type=environment,
                environment_kwargs=filter_sensitive_keys(environment_kwargs)
                if environment_kwargs
                else {},
                other_backends=other_backends,
            )
            if self.logger:
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)

    @contextmanager
    def _spawn_completion_context(self, prompt: str | dict[str, Any]):
        """
        为单个完成调用生成语言模型处理器和环境。
        当上下文退出时清理两者。
        """
        # Create client and wrap in handler
        client: BaseLM = get_client(self.backend, self.backend_kwargs)
        lm_handler = LMHandler(client)

        # Register other clients to be available as sub-call options
        if self.other_backends and self.other_backend_kwargs:
            for backend, kwargs in zip(self.other_backends, self.other_backend_kwargs, strict=True):
                other_client: BaseLM = get_client(backend, kwargs)
                lm_handler.register_client(other_client.model_name, other_client)

        lm_handler.start()

        # Pass handler address to environment so it can make llm_query() calls
        env_kwargs = self.environment_kwargs.copy()
        env_kwargs["lm_handler_address"] = (lm_handler.host, lm_handler.port)
        env_kwargs["context_payload"] = prompt

        # Initialize the environment
        environment: BaseEnv = get_environment(self.environment_type, env_kwargs)

        try:
            yield lm_handler, environment
        finally:
            # Cleanup
            lm_handler.stop()
            if hasattr(environment, "cleanup"):
                environment.cleanup()

    def _setup_prompt(self, prompt: str | dict[str, Any]) -> list[dict[str, Any]]:
        """
        为 RLM 设置系统提示。还包括关于提示的元数据并构建
        初始消息历史记录。
        """
        metadata = QueryMetadata(prompt)
        message_history = build_rlm_system_prompt(
            system_prompt=self.system_prompt, query_metadata=metadata
        )

        return message_history

    def completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        """
        递归语言模型完成调用。这是查询 RLM 的主要入口点，
        可以替代常规的语言模型完成调用。

        为此调用的持续时间生成自己的环境和语言模型处理器。

        参数:
            prompt: 作为上下文传递给模型的单个字符串或消息字典。
            root_prompt: 我们允许 RLM 的根语言模型看到用户指定的（小）提示。一个常见的例子是，
            如果用户要求 RLM 回答问题，我们可以将问题作为根提示传递。
        返回:
            作为字符串的最终答案。
        """
        time_start = time.perf_counter()

        # If we're at max depth, the RLM is an LM, so we fallback to the regular LM.
        if self.depth >= self.max_depth:
            return self._fallback_answer(prompt)

        with self._spawn_completion_context(prompt) as (lm_handler, environment):
            message_history = self._setup_prompt(prompt)

            for i in range(self.max_iterations):
                # Current prompt = message history + additional prompt suffix
                current_prompt = message_history + [build_user_prompt(root_prompt, i)]

                iteration: RLMIteration = self._completion_turn(
                    prompt=current_prompt,
                    lm_handler=lm_handler,
                    environment=environment,
                )

                # Check if RLM is done and has a final answer.
                final_answer = find_final_answer(iteration.response, environment=environment)
                iteration.final_answer = final_answer

                # If logger is used, log the iteration.
                if self.logger:
                    self.logger.log(iteration)

                # Verbose output for this iteration
                self.verbose.print_iteration(iteration, i + 1)

                if final_answer is not None:
                    time_end = time.perf_counter()
                    usage = lm_handler.get_usage_summary()
                    self.verbose.print_final_answer(final_answer)
                    self.verbose.print_summary(i + 1, time_end - time_start, usage.to_dict())
                    return RLMChatCompletion(
                        root_model=self.backend_kwargs.get("model_name", "unknown")
                        if self.backend_kwargs
                        else "unknown",
                        prompt=prompt,
                        response=final_answer,
                        usage_summary=usage,
                        execution_time=time_end - time_start,
                    )

                # Format the iteration for the next prompt.
                new_messages = format_iteration(iteration)

                # Update message history with the new messages.
                message_history.extend(new_messages)

            # Default behavior: we run out of iterations, provide one final answer
            time_end = time.perf_counter()
            final_answer = self._default_answer(message_history, lm_handler)
            usage = lm_handler.get_usage_summary()
            self.verbose.print_final_answer(final_answer)
            self.verbose.print_summary(self.max_iterations, time_end - time_start, usage.to_dict())
            return RLMChatCompletion(
                root_model=self.backend_kwargs.get("model_name", "unknown")
                if self.backend_kwargs
                else "unknown",
                prompt=prompt,
                response=final_answer,
                usage_summary=usage,
                execution_time=time_end - time_start,
            )

    def _completion_turn(
        self,
        prompt: str | dict[str, Any],
        lm_handler: LMHandler,
        environment: BaseEnv,
    ) -> RLMIteration:
        """
        执行 RLM 的单个迭代，包括提示模型
        和代码执行 + 工具执行。
        """
        iter_start = time.perf_counter()
        response = lm_handler.completion(prompt)
        code_block_strs = find_code_blocks(response)
        code_blocks = []

        for code_block_str in code_block_strs:
            code_result: REPLResult = environment.execute_code(code_block_str)
            code_blocks.append(CodeBlock(code=code_block_str, result=code_result))

        iteration_time = time.perf_counter() - iter_start
        return RLMIteration(
            prompt=prompt,
            response=response,
            code_blocks=code_blocks,
            iteration_time=iteration_time,
        )

    def _default_answer(self, message_history: list[dict[str, Any]], lm_handler: LMHandler) -> str:
        """
        如果 RLM 用尽迭代次数且未找到最终答案的默认行为。
        它会使用消息历史记录，并尝试从中生成最终答案。
        """
        current_prompt = message_history + [
            {
                "role": "assistant",
                "content": "Please provide a final answer to the user's question based on the information provided.",
            }
        ]
        response = lm_handler.completion(current_prompt)

        if self.logger:
            self.logger.log(
                RLMIteration(
                    prompt=current_prompt,
                    response=response,
                    final_answer=response,
                    code_blocks=[],
                )
            )

        return response

    def _fallback_answer(self, message: str | dict[str, Any]) -> str:
        """
        如果 RLM 实际上已达到最大深度，应被视为普通语言模型的回退行为。
        """
        client: BaseLM = get_client(self.backend, self.backend_kwargs)
        response = client.completion(message)
        return response
