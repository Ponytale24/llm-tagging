from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import json
from langchain.docstore.document import Document


with open("begriffsdefinitionen_content_tags.json","r",encoding="utf-8") as f:
    definitions = json.load(f)

definitions = [Document(page_content=str(item)) for item in definitions]

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(definitions, embeddings)
db.save_local("utils/definitions_db")