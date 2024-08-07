from typing import List
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings():
    def __init__(self, model):
        self.transformer = SentenceTransformer(model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = []
        for doc in texts:
            # get embedding and convert to list of float, then append to output
            embedded_docs.append(list(self.transformer.encode(doc).astype(float)))
        return embedded_docs
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]