---

<h1 align="center" style="font-size:2.8em">
<span>递归语言模型 (<span style="color:orange">RLM</span>s)</span>
</h1>

<p align="center" style="font-size:1.3em">
  <a href="https://arxiv.org/abs/2512.24601">完整论文</a> •
  <a href="https://alexzhang13.github.io/blog/2025/rlm/">博客文章</a> •
  <a href="https://alexzhang13.github.io/rlm/">文档</a> •
  <a href="https://github.com/alexzhang13/rlm-minimal">RLM 极简版</a>
</p>

<p align="center">
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/style.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/test.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/test.yml/badge.svg" alt="Test" />
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.24601">
    <img src="media/paper_preview.png" alt="论文预览" width="300"/>
  </a>
</p>

## 概述
递归语言模型（RLMs）是一种与任务无关的语言模型（LMs）推理范式，通过使 LM 能够*编程*检查、分解并在其输入上递归调用自身，从而处理近乎无限长度的上下文。RLMs 将标准的 `llm.completion(prompt, model)` 调用替换为 `rlm.completion(prompt, model)` 调用。RLMs 将上下文作为 REPL 环境中的变量，LM 可以与之交互并在其中启动子 LM 调用。

此存储库提供了一个可扩展的推理引擎，用于在标准基于 API 的和本地 LLMs 周围使用 RLMs。最初的实验和想法在 2025 年的一篇[博客文章](https://alexzhang13.github.io/blog/2025/rlm/)中提出，并在一篇[arXiv预印本](https://arxiv.org/abs/2512.24601)中扩展了结果。

> [!NOTE]
> 此存储库包含 RLMs 的推理代码，支持各种沙箱环境。欢迎开源贡献。此存储库由 MIT OASYS 实验室的论文作者维护。

<!-- ## 安装
```
pip install rlm
```
要从 `main` 安装最新版本：
```
pip install git+https://github.com/alexzhang13/rlm.git
```
``` -->

## 快速设置
使用 `uv`（或您选择的虚拟环境）设置依赖项：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init && uv venv --python 3.12  # 根据需要更改版本
uv pip install -e .
```

要运行快速测试，以下命令将使用 OpenAI 客户端运行 RLM 查询，使用您的环境变量 `OPENAI_API_KEY`（您可以随意更改）。这将生成控制台输出以及日志，您可以使用可视化工具探索轨迹。
```bash
uv run examples/quickstart.py
```

默认的 RLM 客户端使用通过 Python `exec` 调用在主机进程上运行的 REPL 环境。它使用与主机进程相同的虚拟环境（即它将有权访问相同的依赖项），但在可用的全局模块方面有一些限制。例如，我们可以使用 GPT-5-nano 调用 RLM 完成：
```python
from rlm import RLM

rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-nano"},
    verbose=True,  # 用于在控制台中使用 rich 打印，默认禁用。
)

print(rlm.completion("Print me the first 100 powers of two, each on a newline.").response)
```

## REPL 环境
我们支持两种类型的 REPL 环境 - 隔离和非隔离。非隔离环境（默认）在与 RLM 相同的机器上运行代码执行（例如通过 `exec`），这对于某些本地低风险任务（如简单基准测试）是相当合理的，但如果提示或工具调用可以与恶意用户交互，则可能会出现问题。完全隔离的环境使用基于云的沙箱（例如 Prime Sandboxes、[Modal Sandboxes](https://modal.com/docs/guide/sandboxes)）来运行 RLM 生成的代码，确保与主机进程完全隔离。可以添加环境，但我们原生支持以下环境：`local`（默认）、`modal`、`prime`。

```python
rlm = RLM(
    environment="...", # "local", "docker", "modal", "prime"
    environment_kwargs={...},
)
```

### 本地环境
默认的 `local` 环境 `LocalREPL` 在与 RLM 本身相同的进程中运行，具有指定的全局和本地命名空间，以实现最小安全性。使用此 REPL 通常是安全的，但不应在生产设置中使用。它还与主机进程共享相同的虚拟环境（例如 Conda 或 uv）。

#### Docker <img src="https://github.com/docker.png" alt="Docker" height="20" style="vertical-align: middle;"/>（*需要[安装 Docker](https://docs.docker.com/desktop/setup/install/)*）
我们还支持名为 `DockerREPL` 的基于 Docker 的环境，该环境将 REPL 环境作为 Docker 镜像启动。默认情况下，我们使用 `python:3.11-slim` 镜像，但用户也可以指定自定义镜像。

### 隔离环境
我们支持在单独的基于云的机器上运行的几种不同的 REPL 环境。每当在这些实例中进行递归子调用时，它会从主机进程请求。

#### Modal Sandboxes <img src="https://github.com/modal-labs.png" alt="Modal" height="20" style="vertical-align: middle;"/>
要使用 [Modal Sandboxes](https://modal.com/docs/guide/sandboxes) 作为 REPL 环境，您需要安装并验证您的 Modal 账户。
```bash
uv add modal  # 添加 modal 库
modal setup   # 验证账户
```

#### Prime Intellect Sandboxes <img src="https://github.com/PrimeIntellect-ai.png" alt="Prime Intellect" height="20" style="vertical-align: middle;"/>
> [!WARNING]
> **Prime Intellect Sandboxes** 目前是测试版功能。有关更多信息，请参阅[文档](https://docs.primeintellect.ai/sandboxes/overview)。

> [!IMPORTANT]
> **Prime Intellect Sandboxes 尚未在 `rlm` 中实现**。此功能目前不可用，直到我们修复一些错误。

请参阅 [Prime CLI 设置说明](https://docs.primeintellect.ai/inference/overview) 了解如何设置。您需要设置 CLI 密钥。
```bash
export PRIME_API_KEY=...
```


### 模型提供商
我们目前支持大多数主要客户端（OpenAI、Anthropic），以及路由器平台（OpenRouter、Portkey、LiteLLM）。对于本地模型，我们建议使用 vLLM（它与 [OpenAI 客户端](https://github.com/alexzhang13/rlm/blob/main/rlm/clients/openai.py) 接口）。要查看或添加对更多客户端的支持，请从 [`rlm/clients/`](https://github.com/alexzhang13/rlm/tree/main/rlm/clients) 开始查看。

## 相关阅读
* **[2025年12月]** [递归语言模型 arXiv](https://arxiv.org/abs/2512.24601)
* **[2025年10月]** [递归语言模型博客文章](https://alexzhang13.github.io/blog/2025/rlm/)

如果您在研究中使用此代码或存储库，请引用：

```bibtex
@misc{zhang2025recursivelanguagemodels,
      title={Recursive Language Models}, 
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2025},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601}, 
}
```

## 可选调试：可视化 RLM 轨迹
我们还提供了一个简单的可视化工具，用于检查和查看 RLM 轨迹的代码、子 LM 和根 LM 调用。要在每次完成调用时保存可以在可视化工具中查看的日志文件（`.jsonl`），初始化 `RLMLogger` 对象并在初始化时将其传递给 `RLM`：
```python
from rlm.logger import RLMLogger
from rlm import RLM

logger = RLMLogger(log_dir="./logs")
rlm = RLM(
    ...
    logger=logger
)
```

要在本地运行可视化工具，我们使用 Node.js 和 shadcn/ui：
```
cd visualizer/
npm run dev        # 默认 localhost:3001
```

您将可以选择保存的 `.jsonl` 文件
<p align="center">
  <img src="media/visualizer.png" alt="RLM 可视化工具示例" width="800"/>
</p>
