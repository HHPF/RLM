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
    max_iterations=3,
    logger=logger,
    verbose=True,
)

print("RLM 中文演示开始...")

# 运行一个简单的中文任务
try:
    print("\n测试：计算1到10的和")
    result = rlm.completion("请计算1到10的和，详细展示计算过程。")
    print("\n测试成功！结果:")
    print(result.response)
    
    print("\n测试：生成简单代码")
    result = rlm.completion("请生成一个Python函数，计算两个数的和，并用2和3测试该函数。")
    print("\n测试成功！结果:")
    print(result.response)
    
    print("\n测试：逻辑推理")
    result = rlm.completion("如果所有的鸟都会飞，而企鹅是鸟，那么企鹅会飞吗？请详细分析推理过程。")
    print("\n测试成功！结果:")
    print(result.response)
    
    print("\n所有测试完成！")
except Exception as e:
    print(f"\n测试过程中遇到错误: {e}")
