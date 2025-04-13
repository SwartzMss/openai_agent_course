import asyncio
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# ========== 环境加载 ==========
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_URL = os.getenv("API_BASE", "https://api.deepseek.com")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

if not API_KEY:
    raise ValueError("请设置 _API_KEY 环境变量")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# ========== 简单输出护栏 ==========
PHONE_REGEX = re.compile(r"\b(400|800|955\d{2,3}|\d{5,11})\b")

@output_guardrail
async def detect_phone_number(
    context: RunContextWrapper, agent: Agent, output: str
) -> GuardrailFunctionOutput:
    print("[模型原始输出]:", repr(output))
    has_number = bool(PHONE_REGEX.search(output))
    return GuardrailFunctionOutput(
        output_info={"contains_phone_number": has_number},
        tripwire_triggered=has_number,
    )

# ========== 代理定义 ==========
agent = Agent(
    name="简单助手",
    instructions="你是一个有帮助的助手，请根据用户问题给出简洁回答。",
    output_type=str,
    output_guardrails=[detect_phone_number],
    model=MODEL_NAME,
)

# ========== 主函数 ==========
async def main():
    try:
        print("📨 测试 1: 数学问题")
        result1 = await Runner.run(agent, "1 + 1 等于几？")
        print("✅ 输出：", result1.final_output)
    except OutputGuardrailTripwireTriggered as e:
        print("❌ 护栏触发：", e.guardrail_result.output.output_info)

    print("\n-----------------------------\n")

    try:
        print("📨 测试 2: 中国银行客服电话是多少？")
        result2 = await Runner.run(agent, "中国银行的客服电话是多少？")
        print("✅ 输出：", result2.final_output)
    except OutputGuardrailTripwireTriggered as e:
        print("🚨 护栏触发（发现电话号码）：", e.guardrail_result.output.output_info)

if __name__ == "__main__":
    # Windows环境下设置事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
