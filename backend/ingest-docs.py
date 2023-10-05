from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import PGEmbedding

POSTGRES_CONNECTION_URL = "postgresql://postgres:postgres@0.0.0.0:5432/postgres"


def ingest_docs():
    loader = DirectoryLoader(
        path="restonomer-docs",
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    docs_transformed = text_splitter.split_documents(docs)

    embedding = OpenAIEmbeddings(chunk_size=200)

    vectorstore = PGEmbedding(
        embedding_function=embedding,
        connection_string=POSTGRES_CONNECTION_URL,
        collection_name="restonomer",
    )

    record_manager = SQLRecordManager(
        f"postgres/restonomer",
        db_url=POSTGRES_CONNECTION_URL,
    )

    record_manager.create_schema()

    index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
    )


if __name__ == "__main__":
    ingest_docs()
