import os
from collections.abc import Generator
from queue import Queue, Empty
from threading import Thread
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.agents import (
    Tool,
    AgentExecutor,
)
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.schema.runnable import RunnableConfig
from langchain.vectorstores import PGEmbedding
from langsmith import Client

POSTGRES_CONNECTION_URL = "postgresql://postgres:postgres@0.0.0.0:5432/postgres"

client = Client()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(callbacks=[run_collector])
run_id = None
feedback_recorded = False


def search(inp: str, callbacks=None) -> list:
    vectorstore = PGEmbedding(
        embedding_function=OpenAIEmbeddings(),
        connection_string=POSTGRES_CONNECTION_URL,
        collection_name="restonomer",
    )

    retriever = vectorstore.as_retriever(search_kwargs=dict(k=3), callbacks=callbacks)

    docs = retriever.get_relevant_documents(inp, callbacks=callbacks)
    return [doc.page_content for doc in docs]


def get_tools():
    langchain_tool = Tool(
        name="Documentation",
        func=search,
        description="Useful for when you need to refer to Restonomer's documentation",
    )
    return [langchain_tool]


def get_agent(llm, *, chat_history: Optional[list] = None):
    chat_history = chat_history or []
    system_message = SystemMessage(
        content=(
            "You are an expert developer tasked answering questions about the Restonomer. "
            "You have access to a Restonomer knowledge bank which you can query. "
            "You should always first query the knowledge bank for information on the concepts in the question. "
            "For example, given the following input question:\n"
            "-----START OF EXAMPLE INPUT QUESTION-----\n"
            "What is the checkpoint config in restonomer? \n"
            "-----END OF EXAMPLE INPUT QUESTION-----\n"
            "Your research flow should be:\n"
            "1. Query your search tool for information on 'Checkpoints' to get as much context as you can about it.\n"
            "2. Then, query your search tool for information on 'Running Checkpoints' to get as much context as you "
            "can about it.\n"
            "3. Answer the question with the context you have gathered."
            "Include CORRECT code snippets in your answer if relevant to the question. If you can't find the answer, "
            "DO NOT make up an answer. Just say you don't know."
            "Answer the following question as best you can:"
        )
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    memory = AgentTokenBufferMemory(
        memory_key="chat_history", llm=llm, max_token_limit=2000
    )

    for msg in chat_history:
        if "question" in msg:
            memory.chat_memory.add_user_message(str(msg.pop("question")))
        if "result" in msg:
            memory.chat_memory.add_ai_message(str(msg.pop("result")))

    tools = get_tools()

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )

    return agent_executor


class QueueCallback(BaseCallbackHandler):
    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()


@app.post("/chat")
async def chat_endpoint(request: Request):
    global run_id, feedback_recorded, trace_url
    run_id = None
    trace_url = None
    feedback_recorded = False
    run_collector.traced_runs = []

    data = await request.json()
    question = data.get("message")
    chat_history = data.get("history", [])
    data.get("conversation_id")

    print("Received question: ", question)

    def stream() -> Generator:
        global run_id, trace_url, feedback_recorded

        q = Queue()
        job_done = object()

        llm = ChatOpenAI(
            model="gpt-4",
            streaming=True,
            temperature=0,
            callbacks=[QueueCallback(q)],
        )

        def task():
            agent = get_agent(llm, chat_history=chat_history)
            agent.invoke(
                {"input": question, "chat_history": chat_history},
                config=runnable_config,
            )
            q.put(job_done)

        t = Thread(target=task)
        t.start()

        content = ""

        while True:
            try:
                next_token = q.get(True, timeout=1)
                if next_token is job_done:
                    break
                content += next_token
                yield next_token
            except Empty:
                continue

        if not run_id and run_collector.traced_runs:
            run = run_collector.traced_runs[0]
            run_id = run.id

    return StreamingResponse(stream())


@app.post("/feedback")
async def send_feedback(request: Request):
    global run_id, feedback_recorded
    if feedback_recorded or run_id is None:
        return {
            "result": "Feedback already recorded or no chat session found",
            "code": 400,
        }
    data = await request.json()
    score = data.get("score")
    client.create_feedback(run_id, "user_score", score=score)
    feedback_recorded = True
    return {"result": "posted feedback successfully", "code": 200}


trace_url = None


@app.post("/get_trace")
async def get_trace(request: Request):
    global run_id, trace_url
    if trace_url is None and run_id is not None:
        trace_url = client.share_run(run_id)
    if run_id is None:
        return {"result": "No chat session found", "code": 400}
    return trace_url if trace_url else {"result": "Trace URL not found", "code": 400}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
