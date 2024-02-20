# Keyword Extraction

## General

This repo can be used to extract keywords from documents using LLMs or embedding similarity (KeyBERT). It was used in PAPER-LINK. The prompts and models are tested with german documents, but could easily be adapted to other languages.

The `prompt.txt` contains the prompt to extract keywords, while the `candidates.txt` holds all candidate keywords.

## Usage

### Prerequisites

Install the project requirements with `conda`

Specify API Keys in a `.env` in the same directory as the `extract_keywords.py` like this:

```
COHERE_API_KEY=<your-api-key>
OPENAI_API_KEY=<your-api-key>
TOGETHER_API_KEY=<your-api-key>

```

Then run:

```bash
python extract_keywords.py
```

It will try to extract keywords in a single LLM-call if the context length of the specified LLM is long enough to fit the entire text. If not it will divide the text into chunks, get keywords from each chunk and then filter out any duplicates. The results are stored in `results` with the current timestamp and the model used.

Open the `extract_keywords.py`, go to line 25 and 27 and specify the model and documents you want to use.

## Metrics

Example Metrics:

```json
{
  "iMINT_Computer_lernen.docx": {
    "extraction_time": "0:00:00.571182",
    "model": "keybert",
    "chunked": false,
    "keywords": []
  },
  "top_secret_urlaubsfotos.docx": {
    "extraction_time": "0:00:00.362619",
    "model": "keybert",
    "chunked": false,
    "keywords": ["wertsch\u00e4tzende kommunikation", "kommunikation", "angst"]
  },
  "Urlaubsbilder_mit_feedback.docx": {
    "extraction_time": "0:00:00.338842",
    "model": "keybert",
    "chunked": false,
    "keywords": ["symptome"]
  }
}
```

The results contain the extraction time, the model used, an information about whether or not the document was chunked (fitted in model context window) and most importantly the keywords.
These results can be viewed by running `streamlit run dashboard.py` from the terminal. The dashboard also gives a quick info which keywords were in the candidates.

## Models

- [`gpt-4`](https://platform.openai.com/docs/models): Advanced model by OpenAi, used via API. You might need to previous payments of at least 1$ to enable access.
- [`gpt-3.5-turbo-16k`](https://platform.openai.com/docs/models): The OpenAI model that is used in free ChatGPT
- [`cohere`](https://docs.cohere.com/docs/models): Offers a free API but with quite low RateLimit
- [`keybert`](https://maartengr.github.io/KeyBERT/)
