import sys
import os
import paperqa
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import paper_qa_utils
from config.constants import ModelsConstants


def main():


    while True:
        if paper_qa_utils.get_user_confirmation("Do you want to embed further papers? (y/n): "):
            while True:
                try:
                    num_papers = int(input("How many papers would you like to embed?: "))
                    start_position = int(input("What position would you like to start from in the library?: "))
                except ValueError:
                    print("Invalid input. Please enter valid numbers for the number of papers and the start position.")
                    continue

                paper_qa_utils.chatgpt_4o_zotero_embedder(num_papers, start_position)

        else:
            query = input("Paper QA Query: ")
            if query.lower == 'exit':
                print('Exiting...')
                return
            response = docs.query(query)
            print(f"Response: {response}")


if __name__ == "__main__":
    main()
