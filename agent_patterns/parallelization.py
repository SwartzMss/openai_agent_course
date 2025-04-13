import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, ItemHelpers, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled

# 加载 .env 文件中的环境变量
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_URL = os.getenv("API_BASE", "https://api.deepseek.com")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

if not API_KEY:
    raise ValueError("请设置 _API_KEY 环境变量")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# 翻译代理（中文指令）
spanish_agent = Agent(
    name="西语翻译代理",
    instructions="请将用户输入的内容翻译成西班牙语。",
    output_type=str,
    model=MODEL_NAME,
)

# 翻译评选代理（中文指令）
translation_picker = Agent(
    name="翻译评选代理",
    instructions=(
        "你将看到用户的一条原始输入，以及三条西班牙语翻译。\n"
        "请从中选出最准确、最自然的一条。\n"
        "请**原样输出**你选择的翻译，不要修改内容，不要添加任何解释说明。"
    ),
    output_type=str,
    model=MODEL_NAME,
)

# 主逻辑入口
async def main():
    msg = input("请输入要翻译成西班牙语的内容：\n")

    # 并行执行三次翻译
    res_1, res_2, res_3 = await asyncio.gather(
        Runner.run(spanish_agent, msg),
        Runner.run(spanish_agent, msg),
        Runner.run(spanish_agent, msg),
    )

    outputs = [
        ItemHelpers.text_message_outputs(res_1.new_items),
        ItemHelpers.text_message_outputs(res_2.new_items),
        ItemHelpers.text_message_outputs(res_3.new_items),
    ]

    translations = "\n\n".join(outputs)
    print("\n三种翻译结果：\n")
    for idx, t in enumerate(outputs, 1):
        print(f"{idx}. {t}")

    # 使用评选代理选出最佳翻译
    best_translation_result = await Runner.run(
        translation_picker,
        f"用户原始输入：{msg}\n\n三条翻译如下：\n{translations}",
    )

    print("\n-----")
    print("最佳翻译：", best_translation_result.final_output)


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
