from __future__ import annotations

import asyncio
from dotenv import load_dotenv
from pathlib import Path
import os

from openai import AsyncOpenAI

from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
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
本示例展示了如何为部分 Runner.run() 调用使用自定义的 LLM 提供方，
而其他调用则继续使用默认的 OpenAI 连接方式。

使用步骤：
1. 创建一个自定义的 AsyncOpenAI 客户端（传入 Base URL 和 API Key）。
2. 创建一个自定义的 ModelProvider，并在其中使用该客户端。
3. 在调用 Runner.run() 时，通过 run_config 参数传入自定义的 ModelProvider 即可。
   如果不传，则使用默认的 OpenAI 连接方式。
"""

# 使用自定义客户端
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# 禁用 tracing
set_tracing_disabled(disabled=True)


class CustomModelProvider(ModelProvider):
    """
    自定义模型提供方，实现 get_model() 方法并返回自定义模型。
    这样一来，就可以使用自定义的 base_url 和 api_key。
    """
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(
            model=model_name or MODEL_NAME,
            openai_client=client
        )


# 实例化自定义提供方
CUSTOM_MODEL_PROVIDER = CustomModelProvider()

async def main():
    """
    主函数：创建 Agent 并调用 Runner。
    当 run_config 中指定了自定义的 CUSTOM_MODEL_PROVIDER 时，使用自定义模型。
    如果不指定，则走默认的 OpenAI 配置。
    """
    agent = Agent(
        name="智能助手",
        instructions="你是一个有帮助的助手，请用中文回答。",
    )

    # 使用自定义 Model Provider
    result = await Runner.run(
        agent,
        "请介绍一下你自己",
        run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER),
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
