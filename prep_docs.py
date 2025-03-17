import os

from dotenv import load_dotenv

from Processors.DocumentProcessor import DocumentProcessor
from Processors.DocumentTokenizationProcessor import DocumentTokenizationProcessor

if __name__ == '__main__':
    load_dotenv(override=True)

    # get all docs under Documents folder
    document_paths = []

    for root, dirs, files in os.walk("Documents"):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in ['.pdf', '.txt', '.md']:
                document_paths.append(os.path.join(root, file))

    doc_processor = DocumentProcessor(document_paths)
    documents = doc_processor.resolve_document()

    print("Documents are resolved successfully.")
    print("Total documents: ", len(documents))
    print("Documents: ", document_paths)

    document_tokenization_processor = DocumentTokenizationProcessor()
    document_tokenization_processor.tokenize(documents)

    print("Documents are tokenized successfully.")
