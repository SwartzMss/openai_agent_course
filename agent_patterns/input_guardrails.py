from __future__ import annotations

import asyncio
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# 加载环境变量
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 获取环境变量配置
BASE_URL = os.getenv("API_BASE", "https://api.deepseek.com")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

if not API_KEY:
    raise ValueError("请设置 _API_KEY 环境变量")

# 设置 OpenAI 客户端
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)


"""
本示例展示如何在 deepseek 环境中使用 Guardrail（护栏机制），
检测输入是否是“数学作业请求”，若是则拒绝回答。
"""

# 护栏代理，用于判断输入是否属于数学作业请求（使用自然语言判断）
guardrail_agent = Agent(
    name="护栏检查代理",
    instructions=(
        "请判断用户是否请求你解答数学题或做数学作业。\n"
        "只返回一行字符串，例如：\n"
        "is_math_question=true\n"
        "或\n"
        "is_math_question=false\n"
        "不要返回任何其他内容，也不要解释说明。"
    ),
    output_type=str,
    model=MODEL_NAME,
)


# 护栏逻辑函数：触发则中断主代理执行
@input_guardrail
async def math_guardrail(
    context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=context.context)

    output = result.final_output.strip()
    #print("[Guardrail原始输出]", repr(output))

    # 转为小写 & 去除空格等，做稳健匹配
    normalized = output.replace(" ", "").lower()

    is_math = normalized == "is_math_question=true"

    return GuardrailFunctionOutput(
        output_info={"raw_output": output},
        tripwire_triggered=is_math,
    )

# 主代理逻辑
async def main():
    agent = Agent(
        name="客服代理",
        instructions="你是一个客户支持代理，负责帮助用户解答问题。",
        input_guardrails=[math_guardrail],
        model=MODEL_NAME,
    )

    input_data: list[TResponseInputItem] = []

    while True:
        user_input = input("请输入消息：")
        input_data.append({
            "role": "user",
            "content": user_input,
        })

        try:
            result = await Runner.run(agent, input_data)
            print(result.final_output)
            input_data = result.to_input_list()  # 更新对话上下文
        except InputGuardrailTripwireTriggered:
            message = "抱歉，我无法帮你做数学作业。"
            print(message)
            input_data.append({
                "role": "assistant",
                "content": message,
            })


if __name__ == "__main__":
    # 设置 Windows 事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
