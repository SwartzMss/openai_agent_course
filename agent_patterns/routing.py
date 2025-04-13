import asyncio
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    TResponseInputItem,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# ========== 加载环境变量 ==========
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

# ========== 子代理：支持多语言 ==========
french_agent = Agent(
    name="french_agent",
    instructions="你只能用法语回答。",
    model=MODEL_NAME,
    output_type=str,
)

spanish_agent = Agent(
    name="spanish_agent",
    instructions="你只能用西班牙语回答。",
    model=MODEL_NAME,
    output_type=str,
)

english_agent = Agent(
    name="english_agent",
    instructions="你只能用英语回答。",
    model=MODEL_NAME,
    output_type=str,
)

# ========== 语言判断代理 ==========
language_detector = Agent(
    name="language_detector",
    instructions=(
        "你将接收用户的一句话，任务是判断它应该由哪个代理来处理。\n"
        "你只能返回以下之一：french_agent、spanish_agent、english_agent。\n"
        "不要解释原因，也不要添加任何标点或额外信息。"
    ),
    model=MODEL_NAME,
    output_type=str,
)

# ========== 代理映射 ==========
AGENT_MAP = {
    "french_agent": french_agent,
    "spanish_agent": spanish_agent,
    "english_agent": english_agent,
}

# ========== 主逻辑 ==========
async def main():
    msg = input("你好！我们支持法语、西班牙语和英语。请问有什么可以帮您？\n")
    
    # 首轮使用 triage_agent 做初始判断
    result = await Runner.run(
        Agent(
            name="triage_agent",
            instructions="请根据语言把问题交给适合的代理。",
            handoffs=[french_agent, spanish_agent, english_agent],
            model=MODEL_NAME,
            output_type=str,
        ),
        [{"content": msg, "role": "user"}],
    )

    # 初始化对话上下文和代理
    agent = result._last_agent
    inputs: list[TResponseInputItem] = result.to_input_list()

    print("\n🤖 AI 回复：\n")
    print(result.final_output)
    print(f"[当前代理: {agent.name}]")

    print("\n-----------------------------------------\n")

    while True:
        user_msg = input("你：")
        if not user_msg.strip():
            continue

        # 👇 用语言判断代理判断这句话该给谁处理
        lang_result = await Runner.run(language_detector, user_msg)
        target_name = lang_result.final_output.strip()
        print(f"[判断目标代理: {target_name}]")

        if target_name != agent.name:
            print(f"🔄 检测到语言变更，切换到 {target_name}")
            agent = AGENT_MAP.get(target_name, english_agent)
            inputs = []  # 清除上下文
        inputs.append({"role": "user", "content": user_msg})

        result = await Runner.run(agent, inputs)
        print("\n🤖 AI 回复：\n")
        print(result.final_output)
        print(f"[当前代理: {agent.name}]")
        inputs = result.to_input_list()

        print("\n-----------------------------------------\n")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
