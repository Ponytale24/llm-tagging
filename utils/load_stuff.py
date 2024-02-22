import os
import json
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain.llms.cohere import Cohere
from langchain_together import Together
import tiktoken
from transformers import AutoTokenizer
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class Retriever():
    def __init__(self):
        # load definitions vector store
        self.faiss_db = FAISS.load_local("utils/definitions_db", embeddings=OpenAIEmbeddings())

    def retrieve(self, query):
        return self.faiss_db.similarity_search_with_score(query, k = 10)



def load_docs(doc_path) -> list: 
    """Loads all word or txt documents in "doc_path" directory as a list of langchain documents
    """
    files = []
    if os.path.isfile(doc_path):
        if doc_path.lower().endswith('.docx'):
            files.append(Docx2txtLoader(doc_path).load())
        elif doc_path.lower().endswith('.txt'):
            files.append(TextLoader(doc_path, encoding="utf-8").load())
    if os.path.isdir(doc_path):
        for doc in os.scandir(doc_path):
            if doc.path.lower().endswith('.docx'):
                files.append(Docx2txtLoader(doc.path).load())
            elif doc.path.lower().endswith('.txt'):
                files.append(TextLoader(doc.path, encoding="utf-8").load())
    return files

def load_metadata(path="learning_object_metadata_LLMTagging.csv"):
    if path.endswith("xlsx"):
        df = pd.read_excel(path)
    if path.endswith("csv"):
        df = pd.read_csv(path)
    df = df[["general_title_","general_description_"]]
    return df.values.tolist()
    


def load_candidates():
    with open("candidates.txt", mode="+r", encoding="utf-8") as file:
        candidates = json.load(file)

    candidates = [word.lower() for word in candidates]
    return candidates

def load_llm(model_name: str):
    if model_name == "gpt-3.5-turbo-0125":
        llm = ChatOpenAI(model=model_name, max_retries=5)
        context_length = 16385
        tokenizer = tiktoken.encoding_for_model(model_name)
    if model_name == "gpt-4":
        llm = ChatOpenAI(model=model_name, max_retries=5)
        context_length = 8192
        tokenizer = tiktoken.encoding_for_model(model_name)        
        
    if model_name == "cohere":
        llm =  Cohere(max_retries=5)
        context_length = 4081
        tokenizer = tiktoken.get_encoding("cl100k_base")

    if model_name == "WizardLM/WizardLM-13B-V1.2":
        llm = Together(
        model=model_name,
        temperature=0.7,
        max_tokens=128,
        top_k=1)

        context_length = 4096
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        llm = Together(
        model=model_name,
        temperature=0.7,
        max_tokens=128,
        top_k=1)

        context_length = 32768
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "zero-one-ai/Yi-34B-Chat":
        llm = Together(
        model=model_name,
        temperature=0.7,
        max_tokens=128,
        top_k=1)

        context_length = 4096
        tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B-Chat")
        	
    return llm, context_length, tokenizer