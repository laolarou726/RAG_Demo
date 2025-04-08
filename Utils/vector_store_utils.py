import os
import string

from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_ollama import OllamaEmbeddings


def create_vector_store_from_env() -> [string, Milvus]:
    model_name = os.getenv("MODEL_NAME")
    milvus_db_name = os.getenv("MILVUS_DB_NAME")
    milvus_collection_name = os.getenv("MILVUS_COLLECTION_NAME")
    milvus_host = os.getenv("MILVUS_HOST")
    milvus_port = os.getenv("MILVUS_PORT")

    embeddings = OllamaEmbeddings(model=model_name)

    vectorstore = Milvus(
        consistency_level="Strong",
        vector_field=["dense", "sparse"],
        builtin_function=BM25BuiltInFunction(),
        embedding_function=embeddings,
        collection_name=milvus_collection_name,
        connection_args={"host": milvus_host, "port": milvus_port, "db_name": milvus_db_name}
    )

    return model_name, vectorstore
