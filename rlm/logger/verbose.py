"""
RLM 的详细打印功能，使用 rich 库。您可以随意修改这个文件 :)
我主要用它来调试，很多内容是基于感觉编写的。

提供控制台输出来调试和理解 RLM 执行过程。
使用 "Tokyo Night" 风格的配色方案。
"""

from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from rlm.core.types import CodeBlock, RLMIteration, RLMMetadata

# ============================================================================
# Tokyo Night 配色方案
# ============================================================================
COLORS = {
    "primary": "#7AA2F7",  # 柔和蓝色 - 标题、头部
    "secondary": "#BB9AF7",  # 柔和紫色 - 强调
    "success": "#9ECE6A",  # 柔和绿色 - 成功、代码
    "warning": "#E0AF68",  # 柔和琥珀色 - 警告
    "error": "#F7768E",  # 柔和红/粉色 - 错误
    "text": "#A9B1D6",  # 柔和灰蓝色 - 常规文本
    "muted": "#565F89",  # 静音灰色 - 次要内容
    "accent": "#7DCFFF",  # 明亮青色 - 强调
    "bg_subtle": "#1A1B26",  # 深色背景
    "border": "#3B4261",  # 边框颜色
    "code_bg": "#24283B",  # 代码背景
}

# Rich 样式
STYLE_PRIMARY = Style(color=COLORS["primary"], bold=True)
STYLE_SECONDARY = Style(color=COLORS["secondary"])
STYLE_SUCCESS = Style(color=COLORS["success"])
STYLE_WARNING = Style(color=COLORS["warning"])
STYLE_ERROR = Style(color=COLORS["error"])
STYLE_TEXT = Style(color=COLORS["text"])
STYLE_MUTED = Style(color=COLORS["muted"])
STYLE_ACCENT = Style(color=COLORS["accent"], bold=True)


def _to_str(value: Any) -> str:
    """安全地将任何值转换为字符串。"""
    if isinstance(value, str):
        return value
    return str(value)


class VerbosePrinter:
    """
    RLM 详细输出的 Rich 控制台打印器。

    显示美观、结构化的输出，展示 RLM 的执行过程：
    - 初始配置面板
    - 每次迭代的响应摘要
    - 代码执行及其结果
    - 对其他模型的子调用
    """

    def __init__(self, enabled: bool = True):
        """
        初始化详细打印器。

        参数:
            enabled: 是否启用详细打印。如果为 False，所有方法都不执行任何操作。
        """
        self.enabled = enabled
        self.console = Console() if enabled else None
        self._iteration_count = 0

    def print_header(
        self,
        backend: str,
        model: str,
        environment: str,
        max_iterations: int,
        max_depth: int,
        other_backends: list[str] | None = None,
    ) -> None:
        """打印初始 RLM 配置头部。"""
        if not self.enabled:
            return

        # 主标题
        title = Text()
        title.append("◆ ", style=STYLE_ACCENT)
        title.append("RLM", style=Style(color=COLORS["primary"], bold=True))
        title.append(" ━ 递归语言模型", style=STYLE_MUTED)

        # 配置表格
        config_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 2),
            expand=True,
        )
        config_table.add_column("key", style=STYLE_MUTED, width=16)
        config_table.add_column("value", style=STYLE_TEXT)
        config_table.add_column("key2", style=STYLE_MUTED, width=16)
        config_table.add_column("value2", style=STYLE_TEXT)

        config_table.add_row(
            "Backend",
            Text(backend, style=STYLE_SECONDARY),
            "Environment",
            Text(environment, style=STYLE_SECONDARY),
        )
        config_table.add_row(
            "Model",
            Text(model, style=STYLE_ACCENT),
            "Max Iterations",
            Text(str(max_iterations), style=STYLE_WARNING),
        )

        if other_backends:
            backends_text = Text(", ".join(other_backends), style=STYLE_SECONDARY)
            config_table.add_row(
                "Sub-models",
                backends_text,
                "Max Depth",
                Text(str(max_depth), style=STYLE_WARNING),
            )
        else:
            config_table.add_row(
                "Max Depth",
                Text(str(max_depth), style=STYLE_WARNING),
                "",
                "",
            )

        # 包装在面板中
        panel = Panel(
            config_table,
            title=title,
            title_align="left",
            border_style=COLORS["border"],
            padding=(1, 2),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def print_metadata(self, metadata: RLMMetadata) -> None:
        """将 RLM 元数据打印为头部。"""
        if not self.enabled:
            return

        model = metadata.backend_kwargs.get("model_name", "unknown")
        other = list(metadata.other_backends) if metadata.other_backends else None

        self.print_header(
            backend=metadata.backend,
            model=model,
            environment=metadata.environment_type,
            max_iterations=metadata.max_iterations,
            max_depth=metadata.max_depth,
            other_backends=other,
        )

    def print_iteration_start(self, iteration: int) -> None:
        """打印新迭代的开始。"""
        if not self.enabled:
            return

        self._iteration_count = iteration

        rule = Rule(
            Text(f" 迭代 {iteration} ", style=STYLE_PRIMARY),
            style=COLORS["border"],
            characters="─",
        )
        self.console.print(rule)

    def print_completion(self, response: Any, iteration_time: float | None = None) -> None:
        """打印完成响应。"""
        if not self.enabled:
            return

        # 带时间的头部
        header = Text()
        header.append("◇ ", style=STYLE_ACCENT)
        header.append("LLM 响应", style=STYLE_PRIMARY)
        if iteration_time:
            header.append(f"  ({iteration_time:.2f}s)", style=STYLE_MUTED)

        # 响应内容
        response_str = _to_str(response)
        response_text = Text(response_str, style=STYLE_TEXT)

        # 粗略计算单词数
        word_count = len(response_str.split())
        footer = Text(f"~{word_count} 词", style=STYLE_MUTED)

        panel = Panel(
            Group(response_text, Text(), footer),
            title=header,
            title_align="left",
            border_style=COLORS["muted"],
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_code_execution(self, code_block: CodeBlock) -> None:
        """打印代码执行详情。"""
        if not self.enabled:
            return

        result = code_block.result

        # 头部
        header = Text()
        header.append("▸ ", style=STYLE_SUCCESS)
        header.append("代码执行", style=Style(color=COLORS["success"], bold=True))
        if result.execution_time:
            header.append(f"  ({result.execution_time:.3f}s)", style=STYLE_MUTED)

        # 构建内容
        content_parts = []

        # 代码片段
        code_text = Text()
        code_text.append("代码:\n", style=STYLE_MUTED)
        code_text.append(_to_str(code_block.code), style=STYLE_TEXT)
        content_parts.append(code_text)

        # 如果存在标准输出
        stdout_str = _to_str(result.stdout) if result.stdout else ""
        if stdout_str.strip():
            stdout_text = Text()
            stdout_text.append("\n输出:\n", style=STYLE_MUTED)
            stdout_text.append(stdout_str, style=STYLE_SUCCESS)
            content_parts.append(stdout_text)

        # 如果存在标准错误（错误）
        stderr_str = _to_str(result.stderr) if result.stderr else ""
        if stderr_str.strip():
            stderr_text = Text()
            stderr_text.append("\n错误:\n", style=STYLE_MUTED)
            stderr_text.append(stderr_str, style=STYLE_ERROR)
            content_parts.append(stderr_text)

        # 子调用摘要
        if result.rlm_calls:
            calls_text = Text()
            calls_text.append(f"\n↳ {len(result.rlm_calls)} 个子调用", style=STYLE_SECONDARY)
            content_parts.append(calls_text)

        panel = Panel(
            Group(*content_parts),
            title=header,
            title_align="left",
            border_style=COLORS["success"],
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_subcall(
        self,
        model: str,
        prompt_preview: str,
        response_preview: str,
        execution_time: float | None = None,
    ) -> None:
        """打印对另一个模型的子调用。"""
        if not self.enabled:
            return

        # 头部
        header = Text()
        header.append("  ↳ ", style=STYLE_SECONDARY)
        header.append("子调用: ", style=STYLE_SECONDARY)
        header.append(_to_str(model), style=STYLE_ACCENT)
        if execution_time:
            header.append(f"  ({execution_time:.2f}s)", style=STYLE_MUTED)

        # 内容
        content = Text()
        content.append("提示: ", style=STYLE_MUTED)
        content.append(_to_str(prompt_preview), style=STYLE_TEXT)
        content.append("\n响应: ", style=STYLE_MUTED)
        content.append(_to_str(response_preview), style=STYLE_TEXT)

        panel = Panel(
            content,
            title=header,
            title_align="left",
            border_style=COLORS["secondary"],
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_iteration(self, iteration: RLMIteration, iteration_num: int) -> None:
        """
        打印完整的迭代，包括响应和代码执行。
        这是打印迭代的主要入口点。
        """
        if not self.enabled:
            return

        # 打印迭代头部
        self.print_iteration_start(iteration_num)

        # 打印 LLM 响应
        self.print_completion(iteration.response, iteration.iteration_time)

        # 打印每个代码块执行
        for code_block in iteration.code_blocks:
            self.print_code_execution(code_block)

            # 打印在此代码块期间进行的任何子调用
            for call in code_block.result.rlm_calls:
                self.print_subcall(
                    model=call.root_model,
                    prompt_preview=_to_str(call.prompt) if call.prompt else "",
                    response_preview=_to_str(call.response) if call.response else "",
                    execution_time=call.execution_time,
                )

    def print_final_answer(self, answer: Any) -> None:
        """打印最终答案。"""
        if not self.enabled:
            return

        # 标题
        title = Text()
        title.append("★ ", style=STYLE_WARNING)
        title.append("最终答案", style=Style(color=COLORS["warning"], bold=True))

        # 答案内容
        answer_text = Text(_to_str(answer), style=STYLE_TEXT)

        panel = Panel(
            answer_text,
            title=title,
            title_align="left",
            border_style=COLORS["warning"],
            padding=(1, 2),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def print_summary(
        self,
        total_iterations: int,
        total_time: float,
        usage_summary: dict[str, Any] | None = None,
    ) -> None:
        """在执行结束时打印摘要。"""
        if not self.enabled:
            return

        # 摘要表格
        summary_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 2),
        )
        summary_table.add_column("metric", style=STYLE_MUTED)
        summary_table.add_column("value", style=STYLE_ACCENT)

        summary_table.add_row("迭代次数", str(total_iterations))
        summary_table.add_row("总时间", f"{total_time:.2f}s")

        if usage_summary:
            total_input = sum(
                m.get("total_input_tokens", 0)
                for m in usage_summary.get("model_usage_summaries", {}).values()
            )
            total_output = sum(
                m.get("total_output_tokens", 0)
                for m in usage_summary.get("model_usage_summaries", {}).values()
            )
            if total_input or total_output:
                summary_table.add_row("输入令牌", f"{total_input:,}")
                summary_table.add_row("输出令牌", f"{total_output:,}")

        # 包装在规则中
        self.console.print()
        self.console.print(Rule(style=COLORS["border"], characters="═"))
        self.console.print(summary_table, justify="center")
        self.console.print(Rule(style=COLORS["border"], characters="═"))
        self.console.print()
