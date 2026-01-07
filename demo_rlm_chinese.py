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
    max_iterations=5,
    logger=logger,
    verbose=True,
)

print("RLM 中文演示开始...")

# 测试用例列表
test_cases = [
    {
        "name": "数学计算",
        "prompt": "请计算1到100的和，详细展示计算过程。"
    },
    {
        "name": "代码生成",
        "prompt": "请生成一个Python函数，判断一个数是否为质数，并用10到20之间的数测试该函数。"
    },
    {
        "name": "文本分析",
        "prompt": "请分析以下文本的主要内容和观点：人工智能技术正在快速发展，它已经在许多领域取得了显著成果。然而，人工智能的发展也带来了一些挑战，如就业问题、隐私保护和伦理 concerns。我们需要在促进技术创新的同时，认真考虑这些问题，制定相应的政策和规范。"
    },
    {
        "name": "逻辑推理",
        "prompt": "如果所有的猫都有尾巴，而汤姆是一只猫，那么汤姆有尾巴吗？请详细分析推理过程。"
    }
]

# 运行测试用例
for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"测试用例 {i}: {test_case['name']}")
    print(f"{'='*60}")
    
    try:
        result = rlm.completion(test_case['prompt'])
        print("\n测试成功！结果:")
        print(result.response)
    except Exception as e:
        print(f"\n测试过程中遇到错误: {e}")

print(f"\n{'='*60}")
print("所有测试用例执行完成！")
print(f"{'='*60}")
