from __future__ import annotations
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

import asyncio
from typing import Any, Literal

from pydantic import BaseModel

from agents import (
    Agent,
    FunctionToolResult,
    ModelSettings,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputFunction,
    ToolsToFinalOutputResult,
    function_tool,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# 在脚本文件所在目录的上一层目录中查找 .env 文件并加载环境变量
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 从环境变量获取配置
BASE_URL = os.getenv("API_BASE", "https://api.deepseek.com")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

# 确保API密钥被正确设置
if not API_KEY:
    raise ValueError("请设置 _API_KEY 环境变量")

# 初始化OpenAI客户端
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# 设置默认OpenAI客户端，并禁用tracing跟踪
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

"""
该示例展示了如何强制代理使用工具。使用 ModelSettings(tool_choice="required") 来强制代理调用工具。

可以通过以下3种方式运行脚本:
1. default：默认行为，工具执行的结果会重新送回模型。
2. first_tool_result：使用第一个工具的执行结果作为最终输出。
3. custom：使用自定义工具处理函数决定最终输出内容。

使用方式：
python agent_patterns/forcing_tool_use.py -t default
python agent_patterns/forcing_tool_use.py -t first_tool
python agent_patterns/forcing_tool_use.py -t custom
"""


# 定义天气信息模型
class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


# 工具函数：模拟获取天气信息（实际开发可调用外部API获取真实数据）
@function_tool
def get_weather(city: str) -> Weather:
    print("[调试信息] 调用 get_weather")
    # 模拟返回固定的天气信息
    return Weather(city=city, temperature_range="14-20C", conditions="晴朗且有风")


# 自定义工具调用结果的处理函数
async def custom_tool_use_behavior(
    context: RunContextWrapper[Any], results: list[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    weather: Weather = results[0].output
    # 使用工具结果创建自定义格式的最终输出
    return ToolsToFinalOutputResult(
        is_final_output=True, final_output=f"{weather.city} 当前天气：{weather.conditions}。"
    )


# 主函数：根据命令行输入确定工具使用行为
# tool_use_behavior 参数用于设置代理调用工具后的处理方式，可以是以下值:
#
# - "run_llm_again": 默认行为，工具调用结果会重新送回模型处理
# - "stop_on_first_tool": 调用一个工具后直接终止流程并输出结果
# - custom_tool_use_behavior: 自定义函数，用于控制如何处理多个工具的调用结果
#
# 该参数的具体值在 main() 函数中根据命令行参数动态决定

# ModelSettings 配置说明：
# model_settings=ModelSettings(...) 用于配置模型的行为
# 
# tool_choice="required" 的作用：
# - 强制要求模型必须调用工具，不能直接输出回答
# - 如果希望模型必须使用提供的工具（如天气查询工具），需要设置此参数
# - 如果设置为 None，则模型可以自由选择是否使用工具
# 
# 此配置在 main() 函数中根据命令行参数动态决定

async def main(tool_use_behavior: Literal["default", "first_tool", "custom"] = "default"):
    # 根据输入确定代理调用工具后的行为
    if tool_use_behavior == "default":
        behavior: Literal["run_llm_again", "stop_on_first_tool"] | ToolsToFinalOutputFunction = (
            "run_llm_again"
        )
    elif tool_use_behavior == "first_tool":
        behavior = "stop_on_first_tool"
    elif tool_use_behavior == "custom":
        behavior = custom_tool_use_behavior

    # 创建代理实例，设置工具调用行为与模型
    agent = Agent(
        name="天气代理",
        instructions="你是一个有用的代理。",
        tools=[get_weather],
        tool_use_behavior=behavior,
        model_settings=ModelSettings(
            tool_choice="required" if tool_use_behavior != "default" else None
        ),
        model=MODEL_NAME,
    )

    # 运行代理，向其输入问题并获取最终输出
    result = await Runner.run(agent, input="东京的天气如何？")
    print(result.final_output)


# 脚本入口：解析命令行参数并执行主函数
if __name__ == "__main__":
    import argparse

    # Windows环境下设置事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tool-use-behavior",
        type=str,
        required=True,
        choices=["default", "first_tool", "custom"],
        help="设置工具调用后的处理行为。",
    )
    args = parser.parse_args()
    asyncio.run(main(args.tool_use_behavior))
