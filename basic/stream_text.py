import asyncio

# ResponseTextDeltaEvent 代表的是流式返回的文本增量事件
from openai.types.responses import ResponseTextDeltaEvent

# 从 agents 包中导入 Agent 和 Runner
from agents import Agent, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled   
from dotenv import load_dotenv  
from pathlib import Path
import os
from openai import AsyncOpenAI

# 加载环境变量
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_URL = os.getenv("API_BASE", "https://api.deepseek.com")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

if not API_KEY:
    raise ValueError("请设置 API_KEY 环境变量")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

async def main():
    # 定义一个简单的 Agent，只有最基本的指令
    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
        model=MODEL_NAME,
    )

    # 使用 run_streamed 以流式方式调用 Agent
    result = Runner.run_streamed(
        agent,
        input="Please tell me 5 jokes."
    )

    # 通过异步迭代获取事件
    async for event in result.stream_events():
        # 我们只关注原始增量文本事件 (raw_response_event)
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            # delta 是每次流式返回的一小段文本
            print(event.data.delta, end="", flush=True)

if __name__ == "__main__":
    # 处理 Windows 平台的事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
