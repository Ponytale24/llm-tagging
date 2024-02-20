# Keyword Extraction

## General

This repo can be used to extract keywords from documents using LLMs or embedding similarity with FAISS. It was used in PAPER-LINK. The prompts and models are tested with german documents, but could easily be adapted to other languages.

## Prerequisites

Install the project requirements from the `requirements.txt`with pip.

Specify API Keys in a `.env` in the same directory as the `extract_keywords.py` like this:

```
OPENAI_API_KEY=<your-api-key>
TOGETHER_API_KEY=<your-api-key>
```

## Usage

The `llm_tagging.py` contains the logic to run tagging with LLM (optionally with retrieval of definitions).

We used a .csv-file to store the document contents, but you could any kind of document loader provided by Langchain. The code needs to be adjusted for this.

It will try to extract keywords in a single LLM-call if the context length of the specified LLM is long enough to fit the entire text. If not it will divide the text into chunks, get keywords from each chunk and then filter out any duplicates. The results are stored in `results` with the current timestamp and the model used.

The `embedding_tagging.py` contains the logic to run tagging with only retrieval.

The `prompts` folder contains the prompt to extract keywords, while the `candidates.txt` holds all candidate keywords.

## Results

Example result:

```json
"Document": {
        "model": "gpt-3.5-turbo-0125",
        "context_length": 16385,
        "model_calls": 1,
        "definitions": "a list of retrieved definitions with relevance",
        "query_length": 2066,
        "chunked": false,
        "extraction_time (s)": "4",
        "keywords": [
          "keyword1",
          "keyword4",
          "keyword6",
          "keyword7",
          "keyword9",
        ]
    }
```

## Models

- [`gpt-3.5-turbo-0125`](https://platform.openai.com/docs/models): The OpenAI model that is used in free ChatGPT
- [`WizardLM-13B-V1.2`](https://huggingface.co/WizardLM/WizardLM-13B-V1.2): Empowering Large Pre-Trained Language Models to Follow Complex Instructions, works quite well for german text
- [`Mixtral-8x7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1): is a pretrained generative Sparse Mixture of Experts. The Mixtral-8x7B outperforms Llama 2 70B on most benchmarks
