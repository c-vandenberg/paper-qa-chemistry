import sys
import os
import paperqa
import pickle
from paperqa.contrib import ZoteroDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import paper_qa_utils
from config.constants import ModelsConstants


def main():
    try:
        with open('data/processed/supervised_learning_gpt_4o_mini.pkl', 'rb') as file:
            docs: paperqa.Docs = pickle.load(file)
            docs.llm = ModelsConstants.GPT_LLM_MODEL
            docs.set_client()
        print("Loaded previously pickled ChatGPT-4o Mini Docs object state")
    except FileNotFoundError:
        docs: paperqa.Docs = paperqa.Docs(llm=ModelsConstants.GPT_LLM_MODEL)
        print("No previously pickled ChatGPT-4o Mini Docs object state found. Starting fresh")

    zotero: ZoteroDB = ZoteroDB(library_type='user')
    while True:
        if paper_qa_utils.get_user_confirmation("Would you like to embed additional papers? (y/n): "):
            try:
                num_papers = paper_qa_utils.get_user_positive_integer(
                    f"Enter the number of papers to embed (Total Zotero database papers: {zotero.num_items()}): "
                )
                start_position = paper_qa_utils.get_user_positive_integer(
                    "Enter the starting point for embedding: "
                )
            except ValueError:
                print("Invalid input. Please enter valid numbers for the number of papers and the start position.")
                continue

            docs: paperqa.Docs = paper_qa_utils.chatgpt_4o_zotero_embedder(docs, num_papers, start_position)

        else:
            while True:
                query: str = input("Paper QA Query: ")
                if query.lower() == 'exit':
                    print('Exiting...')
                    return
                response = docs.query(query)
                print(f"Response: {response.formatted_answer}")


if __name__ == "__main__":
    main()
