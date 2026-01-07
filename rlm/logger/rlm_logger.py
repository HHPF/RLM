"""
RLM 迭代的日志记录器。

将 RLMIteration 数据写入 JSON-lines 文件，用于分析和调试。
"""

import json
import os
import uuid
from datetime import datetime

from rlm.core.types import RLMIteration, RLMMetadata


class RLMLogger:
    """将 RLMIteration 数据写入 JSON-lines 文件的日志记录器。"""

    def __init__(self, log_dir: str, file_name: str = "rlm"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = str(uuid.uuid4())[:8]
        self.log_file_path = os.path.join(log_dir, f"{file_name}_{timestamp}_{run_id}.jsonl")

        self._iteration_count = 0
        self._metadata_logged = False

    def log_metadata(self, metadata: RLMMetadata):
        """将 RLM 元数据作为文件的第一个条目记录。"""
        if self._metadata_logged:
            return

        entry = {
            "type": "metadata",
            "timestamp": datetime.now().isoformat(),
            **metadata.to_dict(),
        }

        with open(self.log_file_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

        self._metadata_logged = True

    def log(self, iteration: RLMIteration):
        """将 RLMIteration 记录到文件中。"""
        self._iteration_count += 1

        entry = {
            "type": "iteration",
            "iteration": self._iteration_count,
            "timestamp": datetime.now().isoformat(),
            **iteration.to_dict(),
        }

        with open(self.log_file_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    @property
    def iteration_count(self) -> int:
        """获取已记录的迭代次数。"""
        return self._iteration_count
