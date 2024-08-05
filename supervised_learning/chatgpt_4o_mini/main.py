import sys
import os
import paperqa
import openai
import pickle
import time
from paperqa.contrib import ZoteroDB
from pathlib import PosixPath
from tqdm import tqdm
from typing import Generator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import llm_utils

GPT_LLM_MODEL = 'gpt-4o-mini'


def main():
    try:
        with open('data/processed/supervised_learning_gpt_4o_mini.pkl', 'rb') as file:
            docs = pickle.load(file)
            docs.llm = GPT_LLM_MODEL
            docs.set_client()
        print("Loaded previously pickled ChatGPT-4o Mini Docs object state")
    except FileNotFoundError:
        docs = paperqa.Docs(llm=GPT_LLM_MODEL)
        print("No previously pickled ChatGPT-4o Mini Docs object state found. Starting fresh")

    zotero: ZoteroDB = ZoteroDB(library_type='user')
    papers: Generator = zotero.iterate()

    for i, paper in enumerate(tqdm(papers, desc="Processing Papers", ncols=100, miniters=1, mininterval=0.5), start=1):
        if paper.key in docs.docnames:
            print(f"Skipping already processed paper {i}: {paper.key}")
            continue

        print(f"Processing paper {i}: {paper.key}")

        paper_content: PosixPath = paper.pdf
        num_tokens: int = llm_utils.calculate_tokens_from_pdf(paper_content, 'gpt-4o-mini')

        print(f"Paper contains {num_tokens} input tokens")

        try:
            docs.add(paper.pdf, docname=paper.key)
        except openai.RateLimitError as e:
            print(f"Rate limit exceeded: {e}. Waiting before retrying...")
            time.sleep(60)  # Wait for 60 seconds before retrying
            continue
        except openai.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

        with open('data/processed/supervised_learning_gpt_4o_mini.pkl', 'wb') as file:
            pickle.dump(docs, file)
        print(f"Saved checkpoint after processing paper {i}.")


if __name__ == "__main__":
    main()
