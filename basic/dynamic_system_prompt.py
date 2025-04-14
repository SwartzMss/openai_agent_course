import asyncio
import random
from typing import Literal

# 从 agents 包中导入基础类和 Runner
from agents import Agent, RunContextWrapper, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled
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

# 自定义上下文类，用于存储“风格”参数
class CustomContext:
    """
    这是一个简单的上下文（context）类，用于在 Agent 中传递额外的状态信息。
    style 可以是三个取值之一：haiku、pirate、robot。
    """
    def __init__(self, style: Literal["haiku", "pirate", "robot"]):
        self.style = style

def custom_instructions(
    run_context: RunContextWrapper[CustomContext], agent: Agent[CustomContext]
) -> str:
    """
    动态生成系统提示（system prompt）的函数。根据上下文中的 style 字段，
    返回不同的指令文本。
    """
    context = run_context.context
    if context.style == "haiku":
        return "请使用俳句（haiku）的形式进行回复。"
    elif context.style == "pirate":
        return "请使用海盗（pirate）的口吻进行回复。"
    else:
        return "请使用机器人（robot）的口吻进行回复，并经常说“beep boop”。"

# 定义一个 Agent，系统提示来自于 custom_instructions
agent = Agent(
    name="聊天代理 (Chat agent)",
    instructions=custom_instructions,
    model=MODEL_NAME,
)

async def main():
    # 从三个风格中随机选择一个
    choice: Literal["haiku", "pirate", "robot"] = random.choice(["haiku", "pirate", "robot"])
    context = CustomContext(style=choice)
    print(f"选择的回复风格: {choice}\n")

    # 用户发送一条消息
    user_message = "给我讲个笑话吧。"
    print(f"用户: {user_message}")
    
    # 使用 Runner.run 来运行该 Agent，传入自定义上下文
    result = await Runner.run(agent, user_message, context=context)

    # 打印最终输出
    print(f"助理: {result.final_output}")

if __name__ == "__main__":
    # 处理 Windows 平台的事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
