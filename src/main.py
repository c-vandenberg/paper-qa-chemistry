import sys
import os
from dotenv import load_dotenv
from paperqa.contrib import ZoteroDB
from src.models.zotero_paper_embedder import ZoteroPaperEmbedder
from utils import prompt_utils, llm_utils
from config.constants import ModelsConstants

load_dotenv()

ZOTERO_LIBRARY_ID: str = os.getenv('ZOTERO_USER_ID')
ZOTERO_API_KEY: str = os.getenv('ZOTERO_API_KEY')


def main():
    zotero: ZoteroDB = ZoteroDB(library_type='user')
    while True:
        llm_model = input(f"Enter the LLM model to use (default is GPT-4o Mini): ")
        if not llm_model.strip():
            llm_model = ModelsConstants.GPT_4o_MINI_LLM_MODEL

        try:
            docs = llm_utils.load_paperqa_doc(
                pkl_file_path=f"../data/processed/paper_qa_"
                              f"{llm_model.lower().replace(' ', '_').replace('-', '_')}.pkl",
                llm_model=llm_model
            )
        except ValueError:
            print(f"{llm_model} is not a valid LLM model")
            continue

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
