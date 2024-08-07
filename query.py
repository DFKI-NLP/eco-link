from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

import argparse

parser = argparse.ArgumentParser("query")
parser.add_argument('query')
args = parser.parse_args()

model_name = "gemma:7b"
embeddings = OllamaEmbeddings(model=model_name)

llm = Ollama(model=model_name)

template = "What is industrial material {material}?"
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

chat_response = chain.invoke({'material': args.query})
db = FAISS.load_local("faiss_index_gemma7b", embeddings, allow_dangerous_deserialization=True)

query_text = '{}\nJahr: {}\nLand: {}'.format(chat_response, '2020', '')
print(query_text)
docs = db.similarity_search_with_score(query_text)
print('Closest matches: \n' + '\n'.join([doc[0].metadata['source'] if 'source' in doc[0].metadata else '' for doc in docs]))
