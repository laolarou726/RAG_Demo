import json

from dotenv import load_dotenv

from Utils.vector_store_utils import create_vector_store_from_env

if __name__ == '__main__':
    load_dotenv(override=True)
    [model_name, vector_store] = create_vector_store_from_env()

    while True:
        # read search query
        search_query = input("Enter search query: ")

        # search the query in the vector store
        results = vector_store.similarity_search(search_query)

        for result in results:
            print(json.dumps({
                "content": result.page_content,
                "metadata": result.metadata
            }, indent=4))