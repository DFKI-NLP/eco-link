from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from embeddings import HuggingFaceEmbeddingModel
import os
import pandas as pd
from langchain_community.retrievers import TavilySearchAPIRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import prompts
from tika import parser

with open('tavily_api_key.txt', 'r') as f:
    api_key = f.readline()
os.environ['TAVILY_API_KEY'] = api_key

class EcoinventRecommendation:

    def __init__(self, model_name="llama3.1:8b"):
        # Initialize the model and retriever
        embeddings = HuggingFaceEmbeddingModel()

        if model_name == 'llama3.1:8b':
            self.llm = Ollama(model=model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                load_in_4bit=True
            )
            pipe = pipeline(
                task='text-generation',
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1000,
                return_full_text=False
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        self.retriever = TavilySearchAPIRetriever(k=5)

        self.db = FAISS.load_local(
            "processing/ecoinvent_index_gemma7b", embeddings, allow_dangerous_deserialization=True
        )

        # Initialize query attributes
        self.set_query()  # Start with a clean state

    def set_query(self,
                  component_name=None,
                  producer=None,
                  material=None,
                  product_description=None,
                  datasheet_content=None,
                  search_query=None,
                  search_results=None):
        """
        Resets and sets the internal state of the class for a new query.
        Parameters must be explicitly named to ensure clarity when resetting or updating the query.
        """
        self.component_name = component_name if component_name is not None else 'unknown'
        self.producer = producer if producer is not None else 'unknown'
        self.material = material if material is not None else 'unknown'
        self.product_description = product_description if product_description is not None else 'unknown'
        self.datasheet_content = datasheet_content if datasheet_content is not None else 'unknown'
        self.search_query = search_query if search_query is not None else 'unknown'
        self.search_results = search_results if search_results is not None else 'unknown'

    def get_search_results(self):
        """
        Perform a web search based on the current state of the attributes.
        Updates the search_query and search_results attributes.
        """
        prompt = PromptTemplate.from_template(prompts.web_query_prompt)
        query_chain = prompt | self.llm

        # Create info_dict dynamically for the prompt
        info_dict = {
            "component_name": self.component_name,
            "producer": self.producer,
            "material": self.material,
            "product_description": self.product_description,
        }

        # Generate the search query
        query = query_chain.invoke(info_dict)
        self.search_query = query

        # Fetch and store search results
        self.search_results = self.retriever.invoke(query)

    def dynamic_prompt_builder(self):
        """
        Build the prompt dynamically based on the current attributes.
        """
        context_parts = []

        if self.search_query != 'unknown' and self.search_results != 'unknown':
            context_parts.append(prompts.search_context.format(
                search_query=self.search_query,
                search_results=self.search_results
            ))
        if self.datasheet_content != 'unknown':
            context_parts.append(prompts.datasheet_context.format(
                component_name=self.component_name,
                datasheet_content=self.datasheet_content
            ))

        base_prompt = (
            prompts.instructions +
            prompts.response_example +
            ''.join(context_parts) +
            prompts.query
        )
        return base_prompt

    def get_matches(self):
        """
        Get matches for the component based on the current state of the attributes.
        Assumes that the web search (if needed) has already been performed by the user.
        """
        # Build the prompt dynamically
        prompt_text = self.dynamic_prompt_builder()
        prompt = PromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm

        # Create info_dict dynamically for the prompt
        info_dict = {
            "component_name": self.component_name,
            "producer": self.producer,
            "material": self.material,
            "product_description": self.product_description,
            "datasheet_content": self.datasheet_content,
            "search_query": self.search_query,
            "search_results": self.search_results,
        }

        # Get response and similarity ranking
        chat_response = chain.invoke(info_dict)
        return chat_response, self.document_similarity_ranking(chat_response)

    def document_similarity_ranking(self, query_text, k=5):
        docs = self.db.similarity_search_with_score(query_text, k=k)
        doc_titles = []
        dists = []
        for doc, dist in docs:
            doc_titles.append(doc.metadata.get('name', ''))
            dists.append(dist)
        return doc_titles, dists
