from rlm import RLM
from rlm.logger import RLMLogger
import json

# 创建日志记录器
logger = RLMLogger(log_dir="./logs")

# 从配置文件读取 DeepSeek API 配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 初始化 RLM，使用 DeepSeek 作为后端模型
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": config['deepseek_model'],
        "api_key": config['deepseek_api_key'],
        "base_url": config['deepseek_base_url']
    },
    environment="local",
    environment_kwargs={},
    max_depth=1,
    max_iterations=10,
    logger=logger,
    verbose=True,
)

print("RLM 初始化成功，开始测试中文任务...")

# 运行中文任务
try:
    # 使用中文提示词，要求计算2的幂并输出
    result = rlm.completion("计算2的0次方到9次方，每个结果单独输出一行。只输出数字，不要其他文本。")
    print("\n测试成功！最终结果:")
    print(result.response)
except Exception as e:
    print(f"\n测试过程中遇到错误: {e}")
