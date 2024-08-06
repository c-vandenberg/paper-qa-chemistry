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
from utils import llm_utils


def get_user_confirmation(prompt: str) -> bool:
    yes_responses = {'y', 'yes'}
    no_responses = {'n', 'no'}

    while True:
        response = input(prompt).strip().lower()
        if response in yes_responses:
            return True
        elif response in no_responses:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def chatgpt_4o_zotero_embedder(embedded_docs, query_limit: int, query_start: int):
    zotero: ZoteroDB = ZoteroDB(library_type='user')
    papers: Generator = zotero.iterate(limit=query_limit, start=query_start)

    for i, paper in enumerate(tqdm(papers, desc="Processing Papers", ncols=100, miniters=1, mininterval=0.5),
                              start=1):
        if paper.key in embedded_docs.docnames:
            print(f"Skipping already processed paper {i}: {paper.key}")
            continue

        print(f"Processing paper {i}: {paper.key}")

        paper_content: PosixPath = paper.pdf
        num_tokens: int = llm_utils.calculate_tokens_from_pdf(paper_content, 'gpt-4o-mini')

        print(f"Paper contains {num_tokens} input tokens")

        try:
            embedded_docs.add(paper.pdf, docname=paper.key)
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

        with open('../chatgpt_4o_mini/data/processed/supervised_learning_gpt_4o_mini.pkl', 'wb') as file:
            pickle.dump(embedded_docs, file)
        print(f"Saved checkpoint after processing paper {i}.")

    return embedded_docs
