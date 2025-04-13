from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from dotenv import load_dotenv

from agents import (
    Agent,
    ItemHelpers,
    Runner,
    TResponseInputItem,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# 加载 .env 环境变量
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_URL = os.getenv("API_BASE", "https://api.deepseek.com")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

if not API_KEY:
    raise ValueError("请设置 _API_KEY 环境变量")

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# 故事大纲生成代理
story_outline_generator = Agent(
    name="故事大纲生成器",
    instructions=(
        "你根据用户输入生成一个非常简短的故事大纲。"
        "如果用户提供了反馈，请结合反馈改进大纲。"
    ),
    model=MODEL_NAME,
)

# 故事评审代理，返回结构化字符串：score=xxx\nfeedback=xxx
evaluator = Agent(
    name="故事大纲评审员",
    instructions=(
        "你负责评审故事大纲是否足够优秀。\n"
        "请只返回如下格式内容，不要添加其他语言：\n"
        "score=pass|needs_improvement|fail\n"
        "feedback=你的建议内容\n"
        "请注意第一次一定不能给出 pass"
    ),
    output_type=str,
    model=MODEL_NAME,
)


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


def parse_feedback(raw_text: str) -> EvaluationFeedback:
    """解析评审员返回的字符串为结构化对象"""
    lines = raw_text.strip().splitlines()
    score = "fail"
    feedback = ""

    for line in lines:
        if line.lower().startswith("score="):
            score = line.split("=", 1)[1].strip()
        elif line.lower().startswith("feedback="):
            feedback = line.split("=", 1)[1].strip()

    return EvaluationFeedback(score=score, feedback=feedback)


# 主函数
async def main() -> None:
    msg = input("你想听一个什么样的故事？")
    input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]
    latest_outline: str | None = None

    while True:
        # 1. 生成故事大纲
        story_outline_result = await Runner.run(
            story_outline_generator,
            input_items,
        )
        input_items = story_outline_result.to_input_list()
        latest_outline = ItemHelpers.text_message_outputs(story_outline_result.new_items)
        print("故事大纲已生成：")
        print(latest_outline)

        # 2. 执行评审逻辑（字符串输出 + 结构化解析）
        evaluator_result = await Runner.run(evaluator, input_items)
        raw_output = evaluator_result.final_output
        print("评审原始输出：\n", raw_output)
        result = parse_feedback(raw_output)

        if result.score == "pass":
            print("大纲已通过评审，流程结束。")
            break

        print("未通过评审，根据反馈重新生成...")
        input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

    print(f"\n最终故事大纲：\n{latest_outline}")


if __name__ == "__main__":
    # Windows环境下设置事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
