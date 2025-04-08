import os

from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

from BGEReranker import BgeCompressor
from Utils.vector_store_utils import create_vector_store_from_env

template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. 

Question: {question} 

Context: {context} 

Answer:
"""


def filter_docs_with_threshold(docs: list, threshold: float):
    filtered = [doc for doc in docs if doc.metadata.get("relevance_score", 1.0) >= threshold]
    return filtered


if __name__ == '__main__':
    load_dotenv(override=True)

    if "COHERE_API_KEY" not in os.environ:
        os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

    [model_name, vector_store] = create_vector_store_from_env()

    prompt = ChatPromptTemplate.from_template(template)
    print(prompt)

    llm = ChatOllama(model=model_name, callbacks=[StreamingStdOutCallbackHandler()])

    reranker = BgeCompressor(model=os.getenv("RERANK_MODEL_NAME"))
    base_retriever = vector_store.as_retriever()

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )


    def retrieve_with_fallback(question: str) -> str:
        # 获取原始检索结果（保留相似度）
        docs: list[Document] = retriever.invoke(question)
        threshold = float(os.getenv("SIMILARITY_THRESHOLD"))
        filtered = filter_docs_with_threshold(docs, threshold)

        if not filtered:
            return os.getenv("FALLBACK_CONTEXT")  # Fallback 模式：没有高质量检索结果
        else:
            # reranker 内部不会再次走向量相似度，可以直接对 filtered 使用 reranker
            reranked_docs = reranker.compress_documents(filtered, question)
            return "\n\n".join([doc.page_content for doc in reranked_docs])


    chain = (
            RunnableParallel({"context": retrieve_with_fallback, "question": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
    )

    while True:
        # read question
        question = input("Chat: ")

        chunks = []
        for chunk in chain.stream(question):
            chunks.append(chunk)

        print()
        print("Response size: ", len(chunks), end='\n')
