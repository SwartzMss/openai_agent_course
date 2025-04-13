import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
from agents import Agent, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled

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

# 第一步：定义生成故事大纲的代理
story_outline_agent = Agent(
    name="story_outline_agent",
    instructions="根据用户的输入生成一个非常简短的故事大纲。",
    model=MODEL_NAME,
)

# 第二步：定义检查故事大纲质量及类别的代理（修改为str类型）
outline_checker_agent = Agent(
    name="outline_checker_agent",
    instructions=(
        "阅读给定的故事大纲，判断大纲质量是否良好，并判断它是否属于科幻故事。"
        "请明确输出，例如：“质量良好，是科幻故事。”或者“质量较差，不是科幻故事。”"
    ),
    output_type=str,
    model=MODEL_NAME,
)

# 第三步：定义根据故事大纲生成故事的代理
story_agent = Agent(
    name="story_agent",
    instructions="根据给定的故事大纲写一篇简短的故事。",
    output_type=str,
    model=MODEL_NAME,
)

async def main():
    input_prompt = input("你想要一个什么样的科幻故事？")

    outline_result = await Runner.run(
        story_outline_agent,
        input_prompt,
    )
    print("已生成故事大纲。")

    outline_checker_result = await Runner.run(
        outline_checker_agent,
        outline_result.final_output,
    )

    result_text = outline_checker_result.final_output
    good_quality = "质量良好" in result_text
    is_scifi = "是科幻故事" in result_text

    if not good_quality:
        print("故事大纲质量不够好，流程终止。")
        return

    if not is_scifi:
        print("故事大纲不是科幻类型，流程终止。")
        return

    print("故事大纲质量很好且属于科幻类型，继续生成完整故事。")

    story_result = await Runner.run(
        story_agent,
        outline_result.final_output,
    )

    print(f"生成的故事如下：\n{story_result.final_output}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
