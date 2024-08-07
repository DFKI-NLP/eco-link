from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm
from langchain_community.embeddings import OllamaEmbeddings

loader = DirectoryLoader("../dbtxt/", show_progress=True, loader_cls=TextLoader)

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
texts = text_splitter.split_documents(documents)

model_name = "gemma:7b"
embeddings = OllamaEmbeddings(model=model_name)

db = FAISS.from_texts([''], embeddings)
for doc in tqdm(texts):
    db.add_documents([doc])
db.save_local("faiss_index_gemma7b")
