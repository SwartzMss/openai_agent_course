from __future__ import annotations

import json
import random
import asyncio
import os
from pathlib import Path


# 从 agents 包中导入相关类和函数
from agents import (
    Agent,
    HandoffInputData,
    Runner,
    function_tool,
    handoff,    
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)
from agents.extensions import handoff_filters

# 使用 python-dotenv 加载本地 .env 文件的环境变量
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 加载环境变量
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 从环境变量获取 DeepSeek 的配置
BASE_URL = os.getenv("API_BASE", "https://api.deepseek.com")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

# 如果未设置 API_KEY，抛出错误
if not API_KEY:
    raise ValueError("请设置 API_KEY 环境变量")

# 初始化客户端并设置为默认
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")

# 禁用 tracing
set_tracing_disabled(disabled=True)

@function_tool
def random_number_tool(max: int) -> int:
    """
    一个工具函数：返回 0 到 max 之间的随机整数。
    """
    return random.randint(0, max)


def spanish_handoff_message_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    """
    当要将对话从 second_agent 移交给 Spanish Assistant 时，会调用本函数进行消息过滤。
    这里演示两步操作：
    1. 使用 remove_all_tools 移除与工具调用相关的对话历史（tool_call, tool_result 等）。
    2. 手动去除对话前两条历史记录，仅做演示用。
    """
    # 先移除所有工具调用的记录
    handoff_message_data = handoff_filters.remove_all_tools(handoff_message_data)

    # 再手动丢弃前两条消息
    history = (
        tuple(handoff_message_data.input_history[2:])
        if isinstance(handoff_message_data.input_history, tuple)
        else handoff_message_data.input_history
    )

    return HandoffInputData(
        input_history=history,
        pre_handoff_items=tuple(handoff_message_data.pre_handoff_items),
        new_items=tuple(handoff_message_data.new_items),
    )


# 第一个 Agent：回答简洁，并提供一个随机数工具函数
first_agent = Agent(
    name="Assistant",
    instructions="Be extremely concise.",
    tools=[random_number_tool],
    model=MODEL_NAME,
)

# 一个只说西班牙语、回答简洁的 Agent
spanish_agent = Agent(
    name="Spanish Assistant",
    instructions="You only speak Spanish and are extremely concise.",
    handoff_description="A Spanish-speaking assistant.",
    model=MODEL_NAME,
)

# 第二个 Agent：平时提供帮助性回答，但若检测到用户说西班牙语，就移交给 Spanish Assistant
second_agent = Agent(
    name="Assistant",
    instructions="Be a helpful assistant. If the user speaks Spanish, handoff to the Spanish assistant.",
    handoffs=[handoff(spanish_agent, input_filter=spanish_handoff_message_filter)],
    model=MODEL_NAME,
)


async def main():
    # 不使用 trace 功能
    
    # 1. 首先给 first_agent 发送一条普通用户消息
    result = await Runner.run(first_agent, input="Hi, my name is Sora.")
    print("Step 1 done")

    # 2. 让 first_agent 再生成一个 0~100 的随机数
    result = await Runner.run(
        first_agent,
        input=result.to_input_list()
        + [{"content": "Can you generate a random number between 0 and 100?", "role": "user"}],
    )
    print("Step 2 done")

    # 3. 转而调用 second_agent，询问纽约市人口
    result = await Runner.run(
        second_agent,
        input=result.to_input_list()
        + [
            {
                "content": "I live in New York City. Whats the population of the city?",
                "role": "user",
            }
        ],
    )
    print("Step 3 done")

    # 4. 让 second_agent 遇到西班牙语时触发移交，并改为流式执行
    #    这样我们就可以在 event 流中读取输出增量
    stream_result = Runner.run_streamed(
        second_agent,
        input=result.to_input_list()
        + [
            {
                "content": "Por favor habla en español. ¿Cuál es mi nombre y dónde vivo?",
                "role": "user",
            }
        ],
    )
    # 这里只是简单地把流事件迭代一遍，没有进一步处理
    async for _ in stream_result.stream_events():
        pass

    print("Step 4 done")

    print("\n===Final messages===\n")

    # 由于刚才发生了移交，所以 spanish_handoff_message_filter 被调用，
    # 导致移除了前两条对话，并清除了工具调用记录。
    # 我们查看 stream_result 的最终消息列表，验证过滤效果。
    for item in stream_result.to_input_list():
        print(json.dumps(item, indent=2))


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
