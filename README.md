# RAG DEMO

![GitHub](https://img.shields.io/github/license/laolarou726/RAG_Demo?logo=github&style=for-the-badge)
![Maintenance](https://img.shields.io/maintenance/yes/2025?logo=diaspora&style=for-the-badge)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/laolarou726/RAG_Demo?style=for-the-badge)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/laolarou726/RAG_Demo?logo=github&style=for-the-badge)
![GitHub repo size](https://img.shields.io/github/repo-size/laolarou726/RAG_Demo?logo=github&style=for-the-badge)

This is a minimal demo project to show the capabilities of a RAG system using `LangChain` and `Milvus`, it contains all
the things you required to build a basic RAG system.

## Before Start

First, make a copy of `.env.sample` and rename it to `.env`, and change any fields need to be changed

Then:

1. Setup the `Milvus` as the vector database
    1. See folder `Milvus`
2. Setup the `Ollama` for the document tokenization and interaction
    1. See [Setup - OllamaEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/ollama/)
    2. See [Ollama](https://ollama.com/)
3. Prep the documents used for RAG and the vector DB
4. Copy all the documents to the `Documents` folder under the project root
5. Run `python prep_doc.py` to prepare the documents for the RAG system
6. You can run `python milvus_search.py` to verify all the documents has been loaded to the vector DB

## Start the demo

Run `python main.py` and type anything you want to ask the RAG system

## Screenshots

### Ask RAG a random question

![a2f22b53e37ee6fb92e43bd67d735e20](https://github.com/user-attachments/assets/bfc7f3e0-bea7-4b3f-9f1f-855f4b78c1fa)

### Vector Database

![fb20250642869db7e0e8dd406774fdd3](https://github.com/user-attachments/assets/e8649171-8950-4693-b2d8-02e4d1cbc72b)
![680b5c08190bfca80f51d1aeef6edff9](https://github.com/user-attachments/assets/af31f243-6b4f-4763-9e63-b0e53f2efc75)

