import asyncio
import random

# 从 agents 包中导入所需的类和函数
from agents import Agent, ItemHelpers, Runner, function_tool, set_default_openai_client, set_default_openai_api, set_tracing_disabled
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

@function_tool
def how_many_jokes() -> int:
    """
    返回一个随机整数，表示要讲多少个笑话。
    """
    return random.randint(1, 10)

async def main():
    # 定义一个简单的 Agent：
    # 1. 会先调用 how_many_jokes 工具，得到一个随机数。
    # 2. 然后按照该随机数来讲笑话。
    agent = Agent(
        name="Joker",
        instructions="First call the `how_many_jokes` tool, then tell that many jokes.",
        tools=[how_many_jokes],
        model=MODEL_NAME,
    )

    # 使用 run_streamed 以流式方式运行该 Agent。
    # result 对象会返回一个 async generator（异步生成器），
    # 可以用 async for 的方式逐步获取事件。
    result = Runner.run_streamed(
        agent,
        input="Hello",
    )

    print("=== Run starting ===")

    async for event in result.stream_events():
        # 这是每次从流中获取到的事件，可根据 event.type 进行分类处理
        if event.type == "raw_response_event":
            # 该类型事件通常表示模型的底层原始增量响应，可忽略或自行使用
            continue
        elif event.type == "agent_updated_stream_event":
            # 当 Agent 的状态变更时触发此事件
            print(f"Agent updated: {event.new_agent.name}")
            continue
        elif event.type == "run_item_stream_event":
            # 这是最常用的事件，表示一次主要的对话或工具调用输出
            if event.item.type == "tool_call_item":
                print("-- Tool was called")  # 表示工具函数被调用了
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")  # 工具函数的返回值
            elif event.item.type == "message_output_item":
                # 这通常是 Agent 给出的自然语言回复
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                # 其他类型可根据需要处理或忽略
                pass

    print("=== Run complete ===")

if __name__ == "__main__":
    # 处理 Windows 平台的事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
