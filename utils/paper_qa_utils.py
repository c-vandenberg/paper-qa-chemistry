import os
import paperqa
import openai
import pickle
import time
from paperqa.contrib import ZoteroDB
from pathlib import PosixPath
from tqdm import tqdm
from typing import Generator
from pyzotero import zotero
from utils import llm_utils

ZOTERO_LIBRARY_ID: str = os.getenv('ZOTERO_USER_ID')

ZOTERO_API_KEY: str = os.getenv('ZOTERO_API_KEY')


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


def get_user_positive_integer(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value >= 0:
                return value
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")


def chatgpt_4o_zotero_embedder(embedded_docs: paperqa.Docs, query_limit: int, query_start: int) -> paperqa.Docs:
    zotero: ZoteroDB = ZoteroDB(library_type='user')
    library_size: int = zotero.num_items()

    if query_start > library_size:
        print(f"Starting position ({query_start}) cannot be larger than Zotero database size ({library_size})")
        return embedded_docs

    start = query_start
    while start < library_size:
        papers: Generator = zotero.iterate(limit=query_limit, start=start, sort='title')
        for i, paper in enumerate(tqdm(papers, desc="Processing Papers", ncols=100, miniters=1, mininterval=0.5),
                                  start=start+1):
            zotero_key = paper.details["key"]
            if zotero_key in embedded_docs.docnames:
                print(f"\nSkipping already processed paper {i}: {paper.key}")
                continue

            print(f"\nProcessing paper {i}: {paper.key}")

            paper_content: PosixPath = paper.pdf
            num_tokens: int = llm_utils.calculate_tokens_from_pdf(paper_content, 'gpt-4o-mini')

            print(f"\nPaper contains {num_tokens} input tokens")

            try:
                embedded_docs.add(paper.pdf, docname=zotero_key)
            except openai.RateLimitError as e:
                print(f"\nRate limit exceeded: {e}. Waiting before retrying...")
                time.sleep(60)  # Wait for 60 seconds before retrying
                continue
            except openai.OpenAIError as e:
                print(f"\nOpenAI API error: {e}")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                break

            with open('../chatgpt_4o_mini/data/processed/supervised_learning_gpt_4o_mini.pkl', 'wb') as file:
                pickle.dump(embedded_docs, file)
            print(f"\nSaved checkpoint after processing paper {i}.")

        start += query_limit

    return embedded_docs
