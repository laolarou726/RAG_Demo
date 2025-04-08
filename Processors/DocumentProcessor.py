import os
import string

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm


class DocumentProcessor:
    def __init__(self, document_paths: [string]):
        self.model_name = os.getenv("SEMANTIC_CHUNKER_MODEL_NAME")
        self.embeddings = OllamaEmbeddings(model=self.model_name)

        self.chunk_size = int(os.getenv("CHUNK_SIZE"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        self.document_paths = document_paths

    # split the document
    def __split_document(self, documents: [Document]) -> [Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(documents)

        print("Splitting documents into chunks...")

        semantic_chunker = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")

        result = []
        for doc in tqdm(docs):
            semantic_chunks = semantic_chunker.create_documents([doc.page_content])
            for chunk in semantic_chunks:
                result.append(chunk)

        return result

    # read PDF file as LangChain document
    def __resolve_pdf(self, path) -> [Document]:
        loader = PyPDFLoader(file_path=path)
        documents = loader.load()

        return self.__split_document(documents)

    # read plain text file as LangChain document
    def __resolve_plaintext(self, path) -> [Document]:
        loader = TextLoader(file_path=path, encoding='utf-8')
        documents = loader.load()

        return self.__split_document(documents)

    def resolve_document(self) -> [[Document]]:
        results: [[Document]] = []

        for document_path in tqdm(self.document_paths):
            ext = os.path.splitext(document_path)[1]

            if ext == '.pdf':
                results.append(self.__resolve_pdf(document_path))
            if ext in ['.txt', '.md']:
                results.append(self.__resolve_plaintext(document_path))

        return results
