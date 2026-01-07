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

print("测试 DeepSeek API 连接...")

# 直接测试 DeepSeek API
try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "Print me the first 10 powers of two, each on a newline."}
        ]
    )
    print("API 调用成功！")
    print("响应结果:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"API 调用失败: {e}")
