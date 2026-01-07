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

print("RLM 初始化成功，开始测试简单任务...")

# 运行一个非常简单的任务
try:
    # 使用更明确的指令，要求直接输出结果
    result = rlm.completion("Calculate 2^0 to 2^9 and print each result on a separate line. Output only the numbers, no extra text.")
    print("\n测试成功！最终结果:")
    print(result.response)
except Exception as e:
    print(f"\n测试过程中遇到错误: {e}")
