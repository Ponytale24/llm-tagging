# imports
# general
import json
import os
import sys
from print_color import print
from time import time
from datetime import timedelta, datetime

from tqdm import tqdm
# utils
from utils.load_stuff import load_metadata, load_candidates, load_llm, Retriever
from utils.parser import MyOutputParser
# llm stuff
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

USE_ALL_MODELS = False
USE_DEFINITIONS = False

available_llms = ["cohere", "gpt-3.5-turbo-0125", "gpt-4", "WizardLM/WizardLM-13B-V1.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "zero-one-ai/Yi-34B-Chat"]
prompt_path = "prompts/main.txt"

if USE_ALL_MODELS:
    models = available_llms
else:
    # options are "cohere", "gpt-3.5-turbo-16k", "gpt-4"  or "WizardLM/WizardLM-13B-V1.2" or "mistralai/Mixtral-8x7B-Instruct-v0.1" or "zero-one-ai/Yi-34B-Chat"
    models = ["gpt-3.5-turbo-0125","WizardLM/WizardLM-13B-V1.2", "mistralai/Mixtral-8x7B-Instruct-v0.1"] 

# loading docs
docs = load_metadata()
# load candidates from txt
candidates = load_candidates()

retriever = Retriever()
# loading api keys
load_dotenv(".env")

folder_name = "results"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")

for model_name in models:
    if model_name not in available_llms:
        print(f"Model {model_name} not found", tag='error', tag_color='red', color='white')
        continue

    print(f"Starting: {model_name}", tag='info', tag_color='blue', color='white')
    # load prompt
    prompt = PromptTemplate.from_file(prompt_path, input_variables=["candidates", "general_title","general_description","definitions"])
    
    llm, max_context_length, tokenizer = load_llm(model_name)

    extract_keywords_chain = prompt | llm | MyOutputParser()

    results = dict()
    with tqdm(total=len(docs), desc="Tagging Content", position=0, leave=True) as pbar:
        for doc in docs:
            title = doc[0]
            pbar.set_description(f"Processing document: {title}", refresh=True)
            description = doc[1]

            if USE_DEFINITIONS:
                definitions_scores = retriever.retrieve(query=title+description)
                str_definitions = str([item[0].page_content + f"(Relevance = {1/item[1]})" for item in definitions_scores])
            else:
                str_definitions = ""

            # extract keywords
            start = time()
            model_calls = 0
            query_length = len(tokenizer.encode(prompt.format(candidates = candidates,general_title=title, general_description=description, definitions=str_definitions)))
            
            if query_length < max_context_length:
                keywords = extract_keywords_chain.invoke({"candidates":candidates,"general_title":title, "general_description":description, "definitions":str_definitions})
                chunked = False
                model_calls += 1
            else: 
                prompt_length = len(tokenizer.encode(prompt.format(candidates = candidates,general_title=title, general_description=description, definitions=str_definitions)))
                chunk_size = max_context_length-prompt_length-500
                try:
                    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(model_name=model_name, chunk_size=chunk_size)
                except:
                    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size)
                split_docs = text_splitter.split_documents(description)
                chunked = True
                keywords_per_chunk = []
                for chunk in split_docs:
                    keywords_per_chunk.append(extract_keywords_chain.invoke({"candidates" : candidates,"general_title":title, "general_description":chunk, "definitions":str_definitions}))
                    model_calls += 1
        
                # creates a list of unique keywords
                keywords = list(set(string for sublist in keywords_per_chunk for string in sublist))

            # TODO should duplicates be removed?
            extraction_time = time() - start
            # Create a timedelta object and format to string 
            extraction_time = str(timedelta(seconds=extraction_time).seconds)
            
            results[doc[0]]={
                "model":model_name,
                "context_length":max_context_length,
                "model_calls": model_calls,
                "definitions": str_definitions if USE_DEFINITIONS else False, 
                "query_length" : query_length,
                "chunked":chunked,
                "extraction_time (s)":extraction_time,
                "keywords":keywords}

            pbar.set_description("", refresh=True)
            pbar.update(1)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')

    # saving metrics to json
    with open(f"results/{timestamp}-{model_name.split('/')[-1]}.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4)
        

