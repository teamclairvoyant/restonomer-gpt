from collections.abc import Generator
from queue import Queue, Empty
from threading import Thread

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

POSTGRES_CONNECTION_URL = "postgresql://postgres:postgres@0.0.0.0:5432/postgres"

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

run_id = None


class QueueCallback(BaseCallbackHandler):
    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()


def search(inp: str, callbacks=None) -> list:
    vectorstore = PGEmbedding(
        embedding_function=OpenAIEmbeddings(),
        connection_string=POSTGRES_CONNECTION_URL,
        collection_name="restonomer",
    )

    retriever = vectorstore.as_retriever(search_kwargs=dict(k=3), callbacks=callbacks)

    return [doc.page_content for doc in retriever.get_relevant_documents(inp, callbacks=callbacks)]


@app.post("/chat")
async def chat_endpoint(request: Request):
    global run_id

    run_id = None

    run_collector.traced_runs = []

    data = await request.json()

    question = data.get("message")
    print("Received question: ", question)

    chat_history = data.get("history", [])

    def stream() -> Generator:
        global run_id

        q = Queue()

        job_done = object()

        llm = ChatOpenAI(
            model="gpt-4",
            streaming=True,
            temperature=0,
            callbacks=[QueueCallback(q)],
        )

        def task():
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
                    "1. Query your search tool for information on 'Checkpoints' to get as much context as you can "
                    "about it.\n"
                    "2. Then, query your search tool for information on 'Running Checkpoints' to get as much context "
                    "as you can about it.\n"
                    "3. Answer the question with the context you have gathered."
                    "Include CORRECT code snippets in your answer if relevant to the question. If you can't find the "
                    "answer,"
                    "DO NOT make up an answer. Just say you don't know."
                    "Answer the following question as best you can:"
                )
            )

            prompt = OpenAIFunctionsAgent.create_prompt(
                system_message=system_message,
                extra_prompt_messages=[
                    MessagesPlaceholder(variable_name="chat_history")
                ],
            )

            memory = AgentTokenBufferMemory(
                memory_key="chat_history", llm=llm, max_token_limit=2000
            )

            for msg in chat_history:
                if "question" in msg:
                    memory.chat_memory.add_user_message(str(msg.pop("question")))
                if "result" in msg:
                    memory.chat_memory.add_ai_message(str(msg.pop("result")))

            tools = [
                Tool(
                    name="Documentation",
                    func=search,
                    description="Useful for when you need to refer to Restonomer's documentation",
                )
            ]

            agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=True,
                return_intermediate_steps=True,
            )

            agent_executor.invoke(
                {"input": question, "chat_history": chat_history},
                config=RunnableConfig(callbacks=[run_collector]),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
