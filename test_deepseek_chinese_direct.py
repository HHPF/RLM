import openai
import json

# 从配置文件读取 DeepSeek API 配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 配置 DeepSeek API
client = openai.OpenAI(
    api_key=config['deepseek_api_key'],
    base_url=config['deepseek_base_url']
)

print("直接测试 DeepSeek API 中文能力...")

# 测试用例
test_cases = [
    {
        "name": "数学计算",
        "prompt": "请计算1到10的和，详细展示计算过程。"
    },
    {
        "name": "代码生成",
        "prompt": "请生成一个Python函数，计算两个数的和，并用2和3测试该函数。"
    },
    {
        "name": "逻辑推理",
        "prompt": "如果所有的鸟都会飞，而企鹅是鸟，那么企鹅会飞吗？请详细分析推理过程。"
    },
    {
        "name": "文本分析",
        "prompt": "请分析以下文本的主要内容：人工智能技术正在快速发展，它已经在许多领域取得了显著成果。然而，人工智能的发展也带来了一些挑战，如就业问题、隐私保护和伦理问题。我们需要在促进技术创新的同时，认真考虑这些问题，制定相应的政策和规范。"
    }
]

# 运行测试用例
for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"测试用例 {i}: {test_case['name']}")
    print(f"{'='*60}")
    print(f"提示词: {test_case['prompt']}")
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个中文助手，所有回复都使用中文。"},
                {"role": "user", "content": test_case['prompt']}
            ]
        )
        print("\n响应结果:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"\n测试过程中遇到错误: {e}")

print(f"\n{'='*60}")
print("所有测试用例执行完成！")
print(f"{'='*60}")
