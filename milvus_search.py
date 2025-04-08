import os

from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever

from BGEReranker import BgeCompressor
from Utils.vector_store_utils import create_vector_store_from_env

if __name__ == '__main__':
    load_dotenv(override=True)

    if "COHERE_API_KEY" not in os.environ:
        os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

    [model_name, vector_store] = create_vector_store_from_env()

    reranker = BgeCompressor(model=os.getenv("RERANK_MODEL_NAME"))
    base_retriever = vector_store.as_retriever()

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    while True:
        # read search query
        search_query = input("Enter search query: ")

        # search the query in the vector store
        results = retriever.invoke(search_query)

        for idx, result in enumerate(results):
            print("=" * 50)
            print(f"Document {idx}")
            print(result)

            print("Metadata: ", result.metadata)
            print("Content: ", result.page_content)

            print("=" * 50)
