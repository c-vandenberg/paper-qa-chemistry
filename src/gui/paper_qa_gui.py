import sys
import tkinter as tk
from tkinter import simpledialog, messagebox
from dotenv import load_dotenv
import os
from paperqa.contrib import ZoteroDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.zotero_paper_embedder import ZoteroPaperEmbedder
from utils import prompt_utils, llm_utils
from config.constants import ModelsConstants

load_dotenv()

ZOTERO_LIBRARY_ID = os.getenv('ZOTERO_USER_ID')
ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')


class PaperQAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Paper QA Interface')

        self.label = tk.Label(root, text="Welcome to Paper QA")
        self.label.pack()

        self.embed_button = tk.Button(root, text="Embed Additional Papers", command=self.embed_papers)
        self.embed_button.pack()

        self.query_button = tk.Button(root, text="Submit Query", command=self.submit_query)
        self.query_button.pack()

        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.quit_button.pack()

    @staticmethod
    def embed_papers():
        zotero = ZoteroDB(library_type='user')

        llm_model = simpledialog.askstring("Input", "Enter the LLM model to use (default is GPT-4o Mini):")
        if not llm_model.strip():
            llm_model = ModelsConstants.GPT_4o_MINI_LLM_MODEL

        try:
            docs = llm_utils.load_paperqa_doc(
                pkl_file_path=f"../data/processed/paper_qa_"
                              f"{llm_model.lower().replace(' ', '_').replace('-', '_')}.pkl",
                llm_model=llm_model
            )
        except ValueError:
            messagebox.showerror("Error", f"{llm_model} is not a valid LLM model")
            return

        num_papers = simpledialog.askinteger(
            "Input", f"Enter the number of papers to embed "
                     f"(Max batch size = 100. Total Zotero database papers = {zotero.num_items()}): "
        )
        start_position = simpledialog.askinteger("Input", "Enter the database starting point for embedding:")

        if num_papers is None or start_position is None:
            messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")
            return

        zotero_paper_embedder = ZoteroPaperEmbedder(
            library_id=ZOTERO_LIBRARY_ID,
            library_type='user',
            api_key=ZOTERO_API_KEY
        )

        zotero_paper_embedder.chatgpt_4o_embedder(
            embedded_docs=docs,
            query_limit=num_papers,
            query_start=start_position
        )

        messagebox.showinfo("Info", "Embedding completed and saved.")

    def submit_query(self):
        llm_model = simpledialog.askstring("Input", "Enter the LLM model to use (default is GPT-4o Mini):")
        if not llm_model.strip():
            llm_model = ModelsConstants.GPT_4o_MINI_LLM_MODEL

        try:
            docs = llm_utils.load_paperqa_doc(
                pkl_file_path=f"../data/processed/paper_qa_"
                              f"{llm_model.lower().replace(' ', '_').replace('-', '_')}.pkl",
                llm_model=llm_model
            )
        except ValueError:
            messagebox.showerror("Error", f"{llm_model} is not a valid LLM model")
            return

        query = simpledialog.askstring("Input", "Paper QA Query:")
        if query and query.lower() == 'exit':
            self.root.quit()
            return

        response = docs.query(query)
        messagebox.showinfo("Response", response.formatted_answer)
