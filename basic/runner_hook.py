import asyncio
import random
from typing import Any

from pydantic import BaseModel

# 从 agents 包中导入所需的类和函数
from agents import (
    Agent,
    RunContextWrapper,
    RunHooks,
    Runner,
    Tool,
    Usage,
    function_tool,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

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

# 自定义一个 RunHooks，用于在 Agent 执行各个阶段记录并打印事件与 Token 用量
class ExampleHooks(RunHooks):
    def __init__(self):
        self.event_counter = 0  # 记录事件发生次数

    def _usage_to_str(self, usage: Usage) -> str:
        """
        将 Usage 中的 requests、input_tokens、output_tokens、total_tokens 转为字符串。
        """
        return (
            f"{usage.requests} requests, "
            f"{usage.input_tokens} input tokens, "
            f"{usage.output_tokens} output tokens, "
            f"{usage.total_tokens} total tokens"
        )

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Agent {agent.name} started. "
            f"Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Agent {agent.name} ended with output {output}. "
            f"Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Tool {tool.name} started. "
            f"Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Tool {tool.name} ended with result {result}. "
            f"Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_handoff(
        self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent
    ) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Handoff from {from_agent.name} to {to_agent.name}. "
            f"Usage: {self._usage_to_str(context.usage)}"
        )

# 实例化上面定义的钩子类
hooks = ExampleHooks()

### 定义两个函数工具（Tool）
@function_tool
def random_number(max: int) -> int:
    """
    生成一个随机数，范围在 0 到 max 之间。
    """
    return random.randint(0, max)

@function_tool
def multiply_by_two(x: int) -> int:
    """
    返回 x 乘以 2 的结果。
    """
    return x * 2

# 注意：DeepSeek 不支持 JSON Schema，因此将 output_type 改为 str

# 定义一个名为 "Multiply Agent" 的 Agent，用于接收数字并将其乘以 2
multiply_agent = Agent(
    name="Multiply Agent",
    instructions="将数字乘以 2，然后返回最终结果。",
    tools=[multiply_by_two],
    output_type=str,    # 改为 str，不再使用 Pydantic 模型
    model=MODEL_NAME,
)

# 定义一个名为 "Start Agent" 的 Agent，用于生成随机数并决定是否移交给 Multiply Agent
start_agent = Agent(
    name="Start Agent",
    instructions=(
        "调用 random_number 生成一个随机数。"
        "如果这个数字是偶数，则直接输出并结束；"
        "如果是奇数，则移交给 Multiply Agent。"
    ),
    tools=[random_number],
    output_type=str,   # 改为 str，不再使用 Pydantic 模型
    handoffs=[multiply_agent],
    model=MODEL_NAME,
)

async def main() -> None:
    # 从用户输入获取一个最大值
    user_input = input("请输入一个最大数字: ")
    # 运行 start_agent，传入输入文本，并附带自定义的 hooks
    await Runner.run(
        start_agent,
        hooks=hooks,
        input=f"生成 0 到 {user_input} 之间的随机数。",
    )

    print("Done!")

if __name__ == "__main__":
    # 处理 Windows 平台的事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
