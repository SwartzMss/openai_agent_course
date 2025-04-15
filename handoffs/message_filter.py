from __future__ import annotations

import json
import random
import asyncio
import os
from pathlib import Path

# 从 agents 包中导入基础功能与类
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
# handoff_filters 提供了一些常用过滤器，用于在移交时对对话历史进行裁剪
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
    返回一个位于 0 到 max 范围内的随机整数。
    """
    return random.randint(0, max)


def spanish_handoff_message_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    """
    当在对话中要从 second_agent 切换到 Spanish Agent 时调用本函数。
    这里演示如何对对话消息进行二次过滤：
    1. 移除所有工具调用相关的历史记录（如 tool_call, tool_result 等）。
    2. 手动去掉对话最前面的两条消息（仅做演示用）。
    """
    # 首先移除任何工具调用相关的消息
    handoff_message_data = handoff_filters.remove_all_tools(handoff_message_data)

    # 再额外去掉最前面的两条消息
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

# 第一个 Agent：回答非常简明，并且可以调用 random_number_tool
first_agent = Agent(
    name="简明助理",
    instructions="请尽量简明扼要回答。",
    tools=[random_number_tool],
    model=MODEL_NAME,  # 使用 deepseek-chat
)

# 一个只用西班牙语回答的 Agent，不调用任何工具
spanish_agent = Agent(
    name="西班牙语助理",
    instructions="你只能使用西班牙语回答，并且需要尽量简短。",
    handoff_description="一个只说西班牙语的助理。",
    model=MODEL_NAME,  # 使用 deepseek-chat 
)

# 第二个 Agent：一般场合下提供帮助性回答，如果发现用户说西班牙语，就移交给西班牙语助理
second_agent = Agent(
    name="热心助理",
    instructions="你是一个热心助理。如果用户说西班牙语，就将对话移交给西班牙语助理。",
    handoffs=[handoff(spanish_agent, input_filter=spanish_handoff_message_filter)],
    model=MODEL_NAME,  # 使用 deepseek-chat
)


async def main():
    # 第一步：将文本发送给第一个 Agent
    result = await Runner.run(first_agent, input="你好，我叫 Sora。")
    print("第 1 步完成")

    # 第二步：再次给第一个 Agent 提问，并让其调用 random_number_tool
    result = await Runner.run(
        first_agent,
        input=result.to_input_list()
        + [{"content": "能生成一个 0 到 100 之间的随机数吗？", "role": "user"}],
    )
    print("第 2 步完成")

    # 第三步：将对话历史交给第二个 Agent，询问关于纽约市的人口
    result = await Runner.run(
        second_agent,
        input=result.to_input_list()
        + [
            {
                "content": "我住在纽约市，你能告诉我这个城市的人口吗？",
                "role": "user",
            }
        ],
    )
    print("第 3 步完成")

    # 第四步：用户用西班牙语提问，触发移交逻辑 -> 切换到西班牙语助理
    result = await Runner.run(
        second_agent,
        input=result.to_input_list()
        + [
            {
                "content": "Por favor habla en español. ¿Cuál es mi nombre y dónde vivo?",
                "role": "user",
            }
        ],
    )
    print("第 4 步完成")

    print("\n=== 最终的消息列表 ===\n")
    # 输出最终的消息列表，用于查看在移交时如何被过滤
    for message in result.to_input_list():
        print(json.dumps(message, indent=2))


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
