from collections.abc import Generator
from queue import Queue, Empty
from threading import Thread

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.gpt4all import GPT4All
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

        callbacks = [QueueCallback(q)]

        llm = GPT4All(
            model="/Users/rahulbhatia/Downloads/llama-2-7b-chat.ggmlv3.q4_0.bin",
            callbacks=callbacks,
            verbose=True,
        )

        def task():
            vectorstore = PGEmbedding(
                embedding_function=OpenAIEmbeddings(),
                connection_string=POSTGRES_CONNECTION_URL,
                collection_name="restonomer",
            )

            retriever = vectorstore.as_retriever(
                search_kwargs=dict(k=3), callbacks=callbacks
            )

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=True,
            )

            qa.invoke(
                {"query": question, "chat_history": chat_history},
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
