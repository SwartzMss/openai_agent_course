import asyncio
from dotenv import load_dotenv
from pathlib import Path
import os

from openai import AsyncOpenAI

from agents import (
    Agent,
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

"""
本示例展示如何为所有请求设置自定义的 LLM 提供方。我们做了三件事：
1. 创建一个自定义的 AsyncOpenAI 客户端（传入 Base URL 和 API Key）。
2. 将此客户端设置为默认客户端，并且不用于 tracing（跟踪）。
3. 将默认的API 设置为 Chat Completions，因为很多 LLM 提供方还不支持老的 Completions。
"""

# 创建自定义的 OpenAI 客户端
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


async def main():
    """
    主函数：创建 Agent 实例，使用上方的默认客户端和 API，
    并调用 Agent 去回答问题。
    """
    agent = Agent(
        name="智能助手",
        instructions="你是一个有帮助的助手，请用中文回答。",
        model=MODEL_NAME,
    )

    # 使用 Runner 让 Agent 回答问题
    result = await Runner.run(agent, "请介绍一下你自己")
    print(result.final_output)


if __name__ == "__main__":
    # 设置 Windows 事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行异步主函数
    asyncio.run(main())
