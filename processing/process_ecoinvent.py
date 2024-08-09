import pandas
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2
from tqdm import tqdm

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from embeddings import HuggingFaceEmbeddingModel

df = pandas.read_excel("Database-Overview-for-ecoinvent-v3.10.xlsx",
                       sheet_name="EN15804 AO")

columns = ['Product UUID',
           'Reference Product Name',
           'CPC Classification',
           'Product Information']
rows = df[columns].drop_duplicates().values

embeddings = HuggingFaceEmbeddingModel()
dimensions = len(embeddings.embed_query("dummy"))
vector_store = FAISS(embedding_function=embeddings,
                     index=IndexFlatL2(dimensions),
                     docstore=InMemoryDocstore(),
                     index_to_docstore_id={},
                     normalize_L2=False
                     )

for row in tqdm(rows):
    doc = Document(
        page_content="Product Name: {}\nProduct Information: {}\nClassification:{}".format(
            row[1],
            row[3],
            row[2]),
        metadata={"UUID": row[0],
                  "name": row[1]}
    )
    vector_store.add_documents([doc], ids=[row[0]])

vector_store.save_local("ecoinvent_index_gemma7b")
