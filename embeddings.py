from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Sentences we want sentence embeddings for
sentences = ["The quick brown fox jumps over the lazy dog.",
             "Now is the time for all good men to come to the aid of their country."]

class HuggingFaceEmbeddingModel(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings
    def embed_query(self, text):
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding
