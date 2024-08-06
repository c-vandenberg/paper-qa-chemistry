import sys
import os
import paperqa
import openai
import pickle
import time
import llm_utils
from paperqa.contrib import ZoteroDB
from pathlib import PosixPath
from tqdm import tqdm
from typing import Generator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.constants import ModelsConstants


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


def chatgpt_4o_zotero_embedder(query_limit: int, query_start):
    try:
        with open('../chatgpt_4o_mini/data/processed/supervised_learning_gpt_4o_mini.pkl', 'rb') as file:
            docs = pickle.load(file)
            docs.llm = ModelsConstants.GPT_LLM_MODEL
            docs.set_client()
        print("Loaded previously pickled ChatGPT-4o Mini Docs object state")
    except FileNotFoundError:
        docs = paperqa.Docs(llm=ModelsConstants.GPT_LLM_MODEL)
        print("No previously pickled ChatGPT-4o Mini Docs object state found. Starting fresh")

    while True:
        proceed = input("Do you want to embed further papers? (yes/no): ").strip().lower()
        if proceed == 'no':
            print("Exiting...")
            break
        elif proceed == 'yes':
            try:
                num_papers = int(input("How many papers would you like to embed?: "))
                start_position = int(input("What position would you like to start from in the library?: "))
            except ValueError:
                print("Invalid input. Please enter valid numbers for the number of papers and the start position.")
                continue

            zotero: ZoteroDB = ZoteroDB(library_type='user')
            papers: Generator = zotero.iterate(limit=query_limit, start=query_start)

            for i, paper in enumerate(tqdm(papers, desc="Processing Papers", ncols=100, miniters=1, mininterval=0.5),
                                      start=1):
                if i < start_position:
                    continue

                if i >= start_position + num_papers:
                    break

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
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
