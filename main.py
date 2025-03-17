from dotenv import load_dotenv
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from Utils.vector_store_utils import create_vector_store_from_env

template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. 

Question: {question} 

Context: {context} 

Answer:
"""

if __name__ == '__main__':
    load_dotenv(override=True)
    [model_name, vector_store] = create_vector_store_from_env()

    prompt = ChatPromptTemplate.from_template(template)
    print(prompt)

    llm = ChatOllama(model=model_name, callbacks=[StreamingStdOutCallbackHandler()])
    retriever = vector_store.as_retriever()

    chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
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
