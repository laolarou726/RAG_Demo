from typing import Sequence, Optional, Any

import torch
import torch.nn.functional as F
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BgeReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(self, query, documents, top_k=5):
        inputs = self.tokenizer(
            [query] * len(documents),
            [doc.page_content for doc in documents],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
            scores = F.sigmoid(logits)  # Normalize to [0, 1]

        # Sort by score
        sorted_indices = torch.argsort(scores, descending=True)
        reranked_docs = []

        for idx in sorted_indices[:top_k]:
            doc = documents[idx]
            score = scores[idx].item()
            doc.metadata["relevance_score"] = round(score, 4)  # Add to metadata
            reranked_docs.append(doc)

        return reranked_docs


class BgeCompressor(BaseDocumentCompressor):
    is_initialized: bool = False
    reranker: Any = None
    model: Optional[str] = "BAAI/bge-reranker-base"
    top_k: Optional[int] = 3

    def setup(self):
        if self.is_initialized:
            return

        self.is_initialized = True
        self.reranker = BgeReranker(model_name=self.model)

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None, ):
        self.setup()
        return self.reranker.rerank(query, documents, self.top_k)
