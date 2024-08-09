from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from embeddings import HuggingFaceEmbeddingModel

import argparse

parser = argparse.ArgumentParser("query")
parser.add_argument('query')
args = parser.parse_args()

embeddings = HuggingFaceEmbeddingModel()

model_name = "gemma:7b"
llm = Ollama(model=model_name)

template = "What is the industrial product or process meant by the following: {material}? Provide the name of the product or process, and then a one-sentence description."
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

chat_response = chain.invoke({'material': args.query})
db = FAISS.load_local("processing/ecoinvent_index_gemma7b", embeddings, allow_dangerous_deserialization=True)

print(chat_response)
docs = db.similarity_search_with_score(chat_response)
print('Closest matches: \n' + '\n'.join([doc[0].metadata['name'] if 'name' in doc[0].metadata else '' for doc in docs]))
