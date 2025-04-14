import asyncio
import random
from typing import Any
from dotenv import load_dotenv
from pathlib import Path
import os

# 从 agents 包中导入所需的类和函数
from agents import (
    Agent,
    AgentHooks,
    RunContextWrapper,
    Runner,
    Tool,
    function_tool,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled
)
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

# 自定义 AgentHooks，用于在 Agent 的生命周期各阶段执行自定义逻辑
class CustomAgentHooks(AgentHooks):
    def __init__(self, display_name: str):
        self.event_counter = 0
        self.display_name = display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        """
        当 Agent 开始执行时调用。
        """
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        """
        当 Agent 执行结束时调用，并打印最终输出。
        """
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended with output {output}")

    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source: Agent) -> None:
        """
        当执行流程从一个 Agent 移交给另一个 Agent 时调用。
        """
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {source.name} handed off to {agent.name}")

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        """
        当 Agent 即将调用某个工具函数时调用。
        """
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started tool {tool.name}")

    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str) -> None:
        """
        当工具函数执行完毕时调用，并打印工具返回结果。
        """
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended tool {tool.name} with result {result}")

@function_tool
def random_number(max: int) -> int:
    """
    生成一个随机数，范围在 0 到 max 之间。
    """
    return random.randint(0, max)

@function_tool
def multiply_by_two(x: int) -> int:
    """
    将输入的整数乘以 2 并返回。
    """
    return x * 2

# 在这里，我们只使用字符串作为输出类型，不再使用 Pydantic 模型，以避免 JSON Schema 相关的问题

# Multiply Agent：接收一个奇数后，使用 multiply_by_two 工具将其翻倍，并返回结果（文本形式）
multiply_agent = Agent(
    name="Multiply Agent",
    instructions=(
        "你收到一个奇数后，调用 multiply_by_two 工具将其翻倍。"
        "然后将结果以字符串形式返回，例如：'结果是 74'。"
    ),
    tools=[multiply_by_two],
    output_type=str,
    hooks=CustomAgentHooks(display_name="Multiply Agent"),
    model=MODEL_NAME,
)

# Start Agent：使用 random_number 工具生成随机数。
# 如果是偶数，直接输出结果并结束；如果是奇数，则移交给 Multiply Agent 做进一步处理。
start_agent = Agent(
    name="Start Agent",
    instructions=(
        "使用 random_number 工具生成一个随机数。"
        "如果生成的是偶数，则输出类似 '偶数：42' 并结束；"
        "如果是奇数，则把该数字移交给 Multiply Agent。"
    ),
    tools=[random_number],
    output_type=str,
    handoffs=[multiply_agent],
    hooks=CustomAgentHooks(display_name="Start Agent"),
    model=MODEL_NAME,
)

async def main() -> None:
    user_input = input("请输入一个最大数值: ")
    await Runner.run(
        start_agent,
        input=f"生成一个 0 到 {user_input} 之间的随机数。",
    )
    print("Done!")

if __name__ == "__main__":
    # 处理 Windows 平台的事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
