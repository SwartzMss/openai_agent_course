import asyncio
from dotenv import load_dotenv
from pathlib import Path
import os

from openai import AsyncOpenAI

from agents import (
    Agent,
    ItemHelpers,
    MessageOutputItem,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

# 在脚本文件所在目录的上一层目录中查找 .env 文件并加载环境变量
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 从环境变量获取配置
BASE_URL = os.getenv("API_BASE", "https://api.deepseek.com")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")


if not API_KEY:
    raise ValueError("请设置 _API_KEY 环境变量")

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# 将此客户端设置为默认客户端，但不用于 tracing
set_default_openai_client(client=client, use_for_tracing=False)

# 将默认的 OpenAI API 设置为 chat_completions
set_default_openai_api("chat_completions")

# 禁用 tracing
set_tracing_disabled(disabled=True)

"""
此示例展示了“agents-as-tools”（代理作为工具）模式。
一个“前线”代理（orchestrator_agent）接收用户输入，然后根据需要调用对应的翻译代理（如西班牙语代理、法语代理、意大利语代理）作为工具。
"""

# 定义各语言的翻译代理
spanish_agent = Agent(
    name="spanish_agent",
    instructions="你将用户的消息翻译成西班牙语。",  # 代理说明
    handoff_description="从英语翻译到西班牙语的翻译代理",  # 用于工具描述
    model=MODEL_NAME,
)

french_agent = Agent(
    name="french_agent",
    instructions="你将用户的消息翻译成法语。",
    handoff_description="从英语翻译到法语的翻译代理",
    model=MODEL_NAME,
)

italian_agent = Agent(
    name="italian_agent",
    instructions="你将用户的消息翻译成意大利语。",
    handoff_description="从英语翻译到意大利语的翻译代理",
    model=MODEL_NAME,
)

# 主调度代理，用于调用上述翻译代理
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "你是一个翻译调度代理。你会使用给定的工具来进行翻译。"
        "如果被要求翻译成多种语言，你需要依次调用对应的工具。"
        "你自己不做翻译，所有翻译都通过提供的工具完成。"
    ),
    model=MODEL_NAME,
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="翻译用户的消息成西班牙语",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="翻译用户的消息成法语",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="翻译用户的消息成意大利语",
        ),
    ],
    )

# 整理/合成代理，用来对翻译结果做检查、汇总
synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="你检查各个翻译结果，如有需要进行纠正，并将结果合并后输出最终回答。",
    model=MODEL_NAME,
)


async def main():
    # 在此处询问用户要翻译的内容以及目标语言
    msg = input("你好，请告诉我需要翻译的内容以及需要翻译到哪些语言？支持西班牙语 法语 意大利语\n")


    # 1. 由调度代理（orchestrator_agent）处理输入并调用翻译代理
    orchestrator_result = await Runner.run(orchestrator_agent, msg)
    print(orchestrator_result.final_output)
    # 显示翻译过程中的中间结果
    for item in orchestrator_result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text:
                print(f"  - 翻译过程: {text}")

    # 2. 将 orchestrator_result 的输出交给合成代理（synthesizer_agent）进行合并/校对
    synthesizer_result = await Runner.run(
        synthesizer_agent,
        orchestrator_result.to_input_list()  # 将翻译结果作为输入
    )

    # 最终输出
    print(f"\n\n最终结果:\n{synthesizer_result.final_output}")


if __name__ == "__main__":
    # 设置 Windows 事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
