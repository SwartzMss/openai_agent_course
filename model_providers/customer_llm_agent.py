import asyncio
from dotenv import load_dotenv
from pathlib import Path
import os

from openai import AsyncOpenAI

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled

# 在脚本文件所在目录的上一层目录中查找 .env 文件并加载环境变量
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 从环境变量获取配置
BASE_URL = os.getenv("API_BASE", "https://api.deepseek.com")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")



if not API_KEY:
    raise ValueError("请设置 _API_KEY 环境变量")

# 创建异步 OpenAI 客户端
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# 禁用跟踪功能
set_tracing_disabled(disabled=True)

async def main():
    """主函数，演示如何使用自定义代理"""
    try:
        # 创建代理实例
        agent = Agent(
            name="智能助手",
            instructions="你是一个有帮助的助手，请用中文回答。",
            model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        )

        # 通过 Runner 给大模型一个中文的提问，让其进行自我介绍
        user_input = "请介绍一下你自己。"
        result = await Runner.run(agent, user_input)
        print(result.final_output)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    # 设置 Windows 事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行异步主函数
    asyncio.run(main())
