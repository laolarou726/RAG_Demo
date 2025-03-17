import os
import string

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

class DocumentProcessor:
    def __init__(self, document_paths: [string]):
        self.chunk_size = int(os.getenv("CHUNK_SIZE"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        self.document_paths = document_paths

    # split the document
    def __split_document(self, documents: [Document]):
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(documents)

        return docs

    # read PDF file as LangChain document
    def __resolve_pdf(self, path) -> [Document]:
        loader = PyPDFLoader(file_path=path)
        documents = loader.load()

        return self.__split_document(documents)

    # read plain text file as LangChain document
    def __resolve_plaintext(self, path) -> [Document]:
        loader = TextLoader(file_path=path)
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