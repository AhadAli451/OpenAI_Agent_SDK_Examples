import os
import chainlit as cl

from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv
import asyncio

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

#Step one provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

#step two: Model 
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

# step three: Config defined at run level
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)
from agents import Agent

agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent]
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello I am a Agent").send()

@cl.on_message
async def handle_massage(message: cl.Message):
    history = cl.user_session.get("history")



    history.append({"role":"user", "content": message.content})
    result = await Runner.run(
    agent,
    input=history,
    run_config=run_config,
)
    history.append({"role": "assistant", "content":result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()