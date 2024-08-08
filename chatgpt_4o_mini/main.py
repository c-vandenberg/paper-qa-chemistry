import sys
import os
import paperqa
import pickle
from paperqa.contrib import ZoteroDB
from ZoteroPaperEmbedder import ZoteroPaperEmbedder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import prompt_utils, llm_utils
from config.constants import ModelsConstants

ZOTERO_LIBRARY_ID: str = os.getenv('ZOTERO_USER_ID')

ZOTERO_API_KEY: str = os.getenv('ZOTERO_API_KEY')

def main():
    zotero: ZoteroDB = ZoteroDB(library_type='user')
    while True:
        docs = llm_utils.load_paperqa_doc(
            pkl_file_path='data/processed/supervised_learning_gpt_4o_mini.pkl',
            llm_model=ModelsConstants.GPT_LLM_MODEL
        )

        if prompt_utils.get_user_confirmation("Would you like to embed additional papers? (y/n): "):
            try:
                num_papers = prompt_utils.get_user_positive_integer(
                    prompt=f"Enter the number of papers to embed (Max batch size = 100. "
                           f"Total Zotero database papers = {zotero.num_items()}): "
                )
                start_position = prompt_utils.get_user_positive_integer(
                    prompt="Enter the database starting point for embedding: "
                )
            except ValueError:
                print("Invalid input. Please enter valid numbers for the number of papers and the start position.")
                continue

            zotero_paper_embedder: ZoteroPaperEmbedder = ZoteroPaperEmbedder(
                library_id=ZOTERO_LIBRARY_ID,
                library_type='user',
                api_key=ZOTERO_API_KEY
            )

            zotero_paper_embedder.chatgpt_4o_embedder(
                embedded_docs=docs,
                query_limit=num_papers,
                query_start=start_position
            )

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
