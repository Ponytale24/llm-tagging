# imports
# general
import ast
import json
from time import time
from datetime import timedelta, datetime

from tqdm import tqdm
# utils
from utils.load_stuff import load_metadata, load_candidates, Retriever
# llm stuff
from dotenv import load_dotenv




prompt_path = "prompts/main.txt"

# loading docs
docs = load_metadata()
# load candidates from txt
candidates = load_candidates()

retriever = Retriever()
# loading api keys
load_dotenv(".env")

model_name = "faiss"
results = dict()
with tqdm(total=len(docs), desc="Tagging Content", position=0, leave=True) as pbar:
    for doc in docs:
        start = time()
        title = doc[0]
        pbar.set_description(f"Processing document: {title}", refresh=True)
        description = doc[1]

        # TODO mehr keywords als 4 nutzen
        definitions_scores = retriever.retrieve(query=title+description)
        keywords = []
        scores = []
        for item in definitions_scores:
            keywords.append(ast.literal_eval(item[0].page_content)["Begriff"])
            scores.append(item[1])

        extraction_time = time() - start
        # Create a timedelta object and format to string 
        extraction_time = str(timedelta(seconds=extraction_time).seconds)
        
        results[doc[0]]={
            "model":model_name,
            "extraction_time (s)":extraction_time,
            "scores": str(scores),
            "keywords":keywords}

        pbar.set_description("", refresh=True)
        pbar.update(1)

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')

# saving metrics to json
with open(f"results/{timestamp}-{model_name.split('/')[-1]}.json", "w", encoding="utf-8") as json_file:
    json.dump(results, json_file, indent=4)
    

