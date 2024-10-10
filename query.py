from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from embeddings import HuggingFaceEmbeddingModel
import getpass
import os
from langchain_community.retrievers import TavilySearchAPIRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline, BitsAndBytesConfig


api_key = ""
os.environ['TAVILY_API_KEY'] = api_key
import argparse

class DocumentRecommendation():

    def __init__(self, model_name=None):
        embeddings = HuggingFaceEmbeddingModel()

        if model_name == 'llama3.1:8b':
            llm = Ollama(model=model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                         trust_remote_code=True,
                                                         load_in_4bit=True)

        self.retriever = TavilySearchAPIRetriever(k=5)

        template = \
        \
"""======================================================

Component "{component_name}" is one of many components used to make a manufactured product.
The product is described as "{product_description}".
{component_name} is made of material "{material}".
The manufacturer of {component_name} is "{producer}".

You must answer correctly.
Provide the generic name of the material production activity in English for the component "{component_name}".
The component name and material may be in a language other than English, so translate it to English to understand the component and material.
Examples of production activities are "aluminium production, primary, ingot" or "acetaldehyde production, ethylene oxidation".
Following this, provide a one-sentence technical description of the material.
If possible, provide a one-sentence technical description of the process.
If unsure, make your best guess.
Do not provide any additional response. Any additional response beyond these items will be considered incorrect.

======================================================

An example of a correctly formatted response:

Industrial activity name: C3 hydrocarbon production, mixture, petroleum refinery operation
Activity information: 
Gaseous mixture of C3-hydrocarbons, yielded from petroleum refinery operation, assumed to consists of 68% propene (also known as propylene or methyl ethylene) and 32% propane
Transformation process of crude oil entering the petroleum refinery ending with refinery products leaving the petroleum refinery.

======================================================

Component "{component_name}" is one of many components used to make a manufactured product.
The product is described as "{product_description}".
The list of all materials and components used to make "{product_description}":

{items}

"""

        prompt = PromptTemplate.from_template(template)
        self.chain = prompt | llm
        self.db = FAISS.load_local("processing/ecoinvent_index_gemma7b", embeddings, allow_dangerous_deserialization=True)

    def get_matches(self, product_description,
                    component_name, producer, material,
                    component_list, producer_list, material_list,
                    web_search=True):
        context = 'No context available'
        if web_search:
            docs = self.retriever.invoke('{} {} {}'.format(component_name, material, producer))
            context = "\n\n".join(doc.page_content for doc in docs)

        chat_response = self.chain.invoke({'product_description': product_description,
                                           'component_name': component_name,
                                           'material': material,
                                           'producer': producer,
                                           'items': '\n'.join(['{}, {}, {}'.format(c,m,p)
                                                               for c,m,p in zip(component_list,material_list, producer_list)])})

        print(chat_response)
        docs = self.db.similarity_search_with_score(chat_response, k=5)
        doc_titles = []
        dists = []
        for doc, dist in docs:
            doc_titles.append(doc.metadata['name'] if 'name' in doc.metadata else '')
            dists.append(dist)
        print(doc_titles)
        return doc_titles, dists, chat_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser("query")
    parser.add_argument('component')
    parser.add_argument('producer')
    args = parser.parse_args()

    recommender = DocumentRecommendation(model_name='llama3.1:8b')
    matches, distances, _ = recommender.get_matches(args.component, args.producer)
    for m, d in zip(matches, distances):
        print('{}: distance={}'.format(m,d))

