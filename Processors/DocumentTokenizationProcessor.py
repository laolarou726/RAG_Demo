import os
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Milvus
from pymilvus import Collection, MilvusException, connections, db, utility

# this class will help to convert the LangChain document to tokens using Ollama Embedding
class DocumentTokenizationProcessor:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME")
        self.drop_old = bool(os.getenv("DROP_OLD"))
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

    def tokenize(self, documents: [[Document]]):
        self.__prep_db_collection()

        flattened_docs = [x for sublist in documents for x in sublist]

        print("Tokenizing documents...")

        Milvus.from_documents(
            documents=flattened_docs,
            embedding=self.embeddings,
            collection_name=self.milvus_collection_name,
            connection_args={"host": self.milvus_host, "port": self.milvus_port, "db_name": self.milvus_db_name},
            drop_old=self.drop_old
        )

        print("Documents are tokenized successfully.")