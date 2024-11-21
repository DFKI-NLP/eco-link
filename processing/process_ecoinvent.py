import pandas
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2
from tqdm import tqdm

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from embeddings import HuggingFaceEmbeddingModel

ecoinvent_excel_file = 'Database-Overview-for-ecoinvent-v3.10.xlsx'
index_file = 'ecoinvent_index'

if __name__ == '__main__':


    df = pandas.read_excel(ecoinvent_excel_file,
                           sheet_name="EN15804 AO")

    columns = ['Product UUID',
               'Activity Name', # production -> activity name
               'CPC Classification',
               'Product Information']
    rows = df[columns].drop_duplicates()
    rows = rows[rows['Activity Name'].str.contains('production')].values

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
            page_content="Activity name: {}\n\n\n\n{}\n\nClassification:{}".format(
                row[1],
                row[3],
                row[2]),
            metadata={"UUID": row[0],
                      "name": row[1]}
        )
        vector_store.add_documents([doc])

    vector_store.save_local(index_file)
