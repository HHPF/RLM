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

print("测试 DeepSeek API 中文响应...")

# 测试中文提示词
try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个中文助手，所有回复都使用中文。"},
            {"role": "user", "content": "计算2的0次方到9次方，每个结果单独输出一行。只输出数字，不要其他文本。"}
        ]
    )
    print("API 调用成功！")
    print("中文响应结果:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"API 调用失败: {e}")
