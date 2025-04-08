import os

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_ollama import OllamaEmbeddings
from pymilvus import Collection, MilvusException, connections, db, utility
from tqdm import tqdm


# this class will help to convert the LangChain document to tokens using Ollama Embedding
class DocumentTokenizationProcessor:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME")
        self.milvus_db_name = os.getenv("MILVUS_DB_NAME")
        self.milvus_collection_name = os.getenv("MILVUS_COLLECTION_NAME")
        self.milvus_host = os.getenv("MILVUS_HOST")
        self.milvus_port = os.getenv("MILVUS_PORT")

        self.embeddings = OllamaEmbeddings(model=self.model_name)

    def __prep_db_collection(self):
        conn = connections.connect(host=self.milvus_host, port=self.milvus_port)

        try:
            existing_databases = db.list_database()
            if self.milvus_db_name in existing_databases:
                print(f"Database '{self.milvus_db_name}' already exists.")

                # Use the database context
                db.using_database(self.milvus_db_name)

                # Drop all collections in the database
                collections = utility.list_collections()
                for collection_name in collections:
                    collection = Collection(name=collection_name)
                    collection.drop()
                    print(f"Collection '{collection_name}' has been dropped.")

                # db.drop_database(self.milvus_db_name)
                # print(f"Database '{self.milvus_db_name}' has been deleted.")
            else:
                print(f"Database '{self.milvus_db_name}' does not exist.")
                database = db.create_database(self.milvus_db_name)
                print(f"Database '{self.milvus_db_name}' created successfully.")
        except MilvusException as e:
            print(f"An error occurred: {e}")
        finally:
            connections.disconnect("default")
            connections.remove_connection("default")

            connections.disconnect(self.milvus_db_name)
            connections.remove_connection(self.milvus_db_name)

    def tokenize(self, documents: [[Document]]):
        self.__prep_db_collection()

        flattened_docs = [x for sublist in documents for x in sublist]

        print("Tokenizing documents...")

        for doc in tqdm(flattened_docs):
            Milvus.from_documents(
                documents=[doc],
                embedding=self.embeddings,
                builtin_function=BM25BuiltInFunction(),
                connection_args={
                    "host": self.milvus_host,
                    "port": self.milvus_port,
                    "db_name": self.milvus_db_name
                },
                vector_field=["dense", "sparse"],
                collection_name=self.milvus_collection_name,
                drop_old=False,
                consistency_level="Strong"
            )

        print("Documents are tokenized successfully.")
