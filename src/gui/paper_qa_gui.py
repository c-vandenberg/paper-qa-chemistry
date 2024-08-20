import sys
import PySimpleGUI as sg
from dotenv import load_dotenv
import os
from paperqa.contrib import ZoteroDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.zotero_paper_embedder import ZoteroPaperEmbedder
from utils import llm_utils
from config.constants import ModelsConstants

load_dotenv()

ZOTERO_LIBRARY_ID = os.getenv('ZOTERO_USER_ID')
ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')


class PaperQAGUI:
    def __init__(self):
        self.zotero = ZoteroDB(library_type='user')
        self.layout = [
            [sg.Text('Paper QA Interface')],
            [sg.Text('Enter the LLM model to use: ')],
            [sg.InputText(default_text='GPT-4o Mini', key='llm_model_input', size=(40, 1))],
            [sg.Text(f"Enter the number of papers to embed (Max batch size = 100. "
                     f"Total Zotero database papers = {self.zotero.num_items()}): ")],
            [sg.InputText(key='num_papers_input', size=(40, 1), enable_events=True)],
            [sg.Text('Enter the database starting point for embedding')],
            [sg.InputText(key='start_position_input', size=(40, 1), enable_events=True)],
            [sg.Text('Embedding Output: ')],
            [sg.Multiline(size=(80, 20), key='console_multiline', autoscroll=True, disabled=True)],
            [sg.InputText(key='query_input', size=(40, 1)), sg.Text('Paper QA Query')],
            [sg.Button('Embed Additional Papers')],
            [sg.Button('Submit Query')],
            [sg.Button('Exit')],
        ]
        self.window = sg.Window('Paper QA', self.layout)

    def embed_papers(self, llm_model, num_papers, start_position):
        if not llm_model.strip():
            llm_model = ModelsConstants.GPT_4o_MINI_LLM_MODEL

        try:
            docs = llm_utils.load_paperqa_doc(
                pkl_file_path=f"../data/processed/paper_qa_"
                              f"{llm_model.lower().replace(' ', '_').replace('-', '_')}.pkl",
                llm_model=llm_model
            )
        except ValueError:
            sg.popup_error(f"{llm_model} is not a valid LLM model")
            return

        if not num_papers or not start_position:
            sg.popup_error("Invalid input. Please enter valid numbers.")
            return

        zotero_paper_embedder = ZoteroPaperEmbedder(
            library_id=ZOTERO_LIBRARY_ID,
            library_type='user',
            api_key=ZOTERO_API_KEY,
            console_multiline=self.window['console_multiline'],
            window=self.window
        )

        zotero_paper_embedder.chatgpt_4o_embedder(
            embedded_docs=docs,
            query_limit=int(num_papers),
            query_start=int(start_position)
        )

        sg.popup('Embedding completed and saved.')

    def submit_query(self, llm_model, query):
        if not llm_model.strip():
            llm_model = ModelsConstants.GPT_4o_MINI_LLM_MODEL

        try:
            docs = llm_utils.load_paperqa_doc(
                pkl_file_path=f"../data/processed/paper_qa_"
                              f"{llm_model.lower().replace(' ', '_').replace('-', '_')}.pkl",
                llm_model=llm_model
            )
        except ValueError:
            sg.popup_error(f"{llm_model} is not a valid LLM model")
            return

        if query.lower() == 'exit':
            self.window.close()
            return

        response = docs.query(query)
        sg.popup_scrolled(f"Response: {response.formatted_answer}", title="Query Result", size=(50, 20))

    @staticmethod
    def validate_positive_integer(value, field_name):
        if not value.isdigit() or int(value) < 0:
            sg.popup_error(f"Please enter a valid positive integer for {field_name}.")
            return False
        return True

    def run(self):
        while True:
            event, values = self.window.read()
            sg.theme('DarkBlue3')

            if event == sg.WIN_CLOSED or event == 'Exit':
                break

            if event == 'num_papers_input':
                if not self.validate_positive_integer(values['num_papers_input'],
                                                      "the number of papers to embed"):
                    self.window['num_papers_input'].update('')

            if event == 'start_position_input':
                if not self.validate_positive_integer(values['start_position_input'],
                                                      "the database starting point"):
                    self.window['start_position_input'].update('')

            if event == 'Embed Additional Papers':
                self.embed_papers(values['llm_model_input'], values['num_papers_input'], values['start_position_input'])

            if event == 'Submit Query':
                self.submit_query(values['llm_model_input'], values['query_input'])

        self.window.close()
