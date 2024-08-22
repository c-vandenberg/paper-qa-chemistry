import os
import sys
import PySimpleGUI as sg
import paperqa
from dotenv import load_dotenv
from paperqa.contrib import ZoteroDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.zotero_paper_embedder import ZoteroPaperEmbedder
from config.constants import ModelsConstants

load_dotenv()

ZOTERO_LIBRARY_ID = os.getenv('ZOTERO_USER_ID')
ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')


class PaperQAGUI:
    """
    A graphical user interface (GUI) for embedding papers and querying an embedded document set using PySimpleGUI.

    This class provides a simple interface for interacting with a set of academic papers stored in a Zotero
    database. It allows users to embed additional papers into a document set and submit queries against the
    document set, with the option to specify the language model to use for processing the documents.

    Attributes
    ----------
    zotero : ZoteroDB
        An instance of ZoteroDB used to interact with the Zotero database.
    layout : list
        The layout of the PySimpleGUI window, including input fields, buttons, and text outputs.
    window : sg.Window
        The main window of the GUI.

    Methods
    -------
    embed_papers(llm_model: str, num_papers: str, start_position: str)
        Embeds additional papers into the document set using the specified language model.
    submit_query(llm_model: str, query: str)
        Submits a query to the document set and displays the response in a popup window.
    run()
        Runs the main event loop for the GUI, handling user interactions.
    validate_positive_integer(value, field_name)
        Validates that the provided value is a positive integer.
    """
    def __init__(self):
        """
        Initializes the PaperQAGUI class by setting up the PySimpleGUI layout and window.

        The layout includes input fields for specifying the language model, the number of papers to embed,
        and the starting position for embedding, as well as buttons for embedding papers, submitting queries,
        and exiting the application.
        """
        self.zotero = ZoteroDB(library_type='user')
        self.layout = [
            [sg.Text('Paper QA Interface')],
            [sg.Text('Enter the language model to use: ')],
            [sg.InputText(default_text='gpt-4o-mini', key='llm_model_input', size=(40, 1), expand_x=True)],
            [sg.Text(f"Enter the number of papers to embed (Max batch size = 100. "
                     f"Total Zotero database papers = {self.zotero.num_items()}): ")],
            [sg.InputText(key='num_papers_input', size=(40, 1), enable_events=True, expand_x=True)],
            [sg.Text('Enter the database starting point for embedding: ')],
            [sg.InputText(key='start_position_input', size=(40, 1), enable_events=True, expand_x=True)],
            [sg.Text('Embedding Output: ')],
            [sg.Multiline(size=(80, 20), key='console_multiline', autoscroll=True, disabled=True, expand_x=True)],
            [sg.Text('Paper QA Query')],
            [sg.InputText(key='query_input', size=(40, 1), expand_x=True)],
            [sg.Button('Embed Additional Papers')],
            [sg.Button('Submit Query: ')],
            [sg.Button('Exit')],
        ]
        self.window = sg.Window('Paper QA', self.layout, resizable=True, finalize=True)
        self.zotero_paper_embedder = ZoteroPaperEmbedder(
            library_id=ZOTERO_LIBRARY_ID,
            library_type='user',
            api_key=ZOTERO_API_KEY,
            console_multiline=self.window['console_multiline'],
            window=self.window
        )

    def embed_papers(self, llm_model: str, num_papers: str, start_position: str):
        """
        Embeds additional papers into a set of vectors using the specified language model.

        Based on the specified LLM, this method loads a document set from a pickle file or creates a new one if the
        file does not exist. It then embeds a specified number of papers from the Zotero database, starting at a given
        position.

        Parameters
        ----------
        llm_model : str
            The language model to use for processing the documents.
        num_papers : str
            The number of papers to embed into the document set.
        start_position : str
            The starting position in the Zotero database from which to begin embedding papers.

        Notes
        -----
        If the input values for `num_papers` or `start_position` are invalid, an error message is displayed.
        The method updates the GUI to output progress messages to the console.
        """
        if not llm_model.strip():
            llm_model = ModelsConstants.GPT_4o_MINI_LLM_MODEL

        try:
            docs: paperqa.Docs = self.zotero_paper_embedder.load_paperqa_doc(
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

        self.zotero_paper_embedder.embed_docs(
            embedded_docs=docs,
            query_limit=int(num_papers),
            query_start=int(start_position)
        )

        sg.popup('Embedding completed and saved.')

    def submit_query(self, llm_model: str, query: str):
        """
        Submits a query which is then embedded into a vector. This vector is then used to search and summarise the top
        passages in the embedded papers, and the LLM is used to score and select the relevant summaries. The response
        and references are then displayed in a popup window.

        This method loads a document set from a pickle file or creates a new one if the file does not exist.

        Parameters
        ----------
        llm_model : str
            The language model to use for processing the documents.
        query : str
            The query to submit to the document set.

        Notes
        -----
        If the query is "exit", the window is closed.
        """
        if not llm_model.strip():
            llm_model: str = ModelsConstants.GPT_4o_MINI_LLM_MODEL

        try:
            docs: paperqa.Docs = self.zotero_paper_embedder.load_paperqa_doc(
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

        response: paperqa.Answer = docs.query(query)
        sg.popup_scrolled(f"Response: {response.formatted_answer}", title="Query Result", size=(50, 20))

    def run(self):
        """
       Runs the main event loop for the GUI, handling user interactions.

       This method starts the GUI event loop, allowing the user to interact with the interface. It handles
       events such as embedding papers, submitting queries, and validating input fields.
       """
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

    @staticmethod
    def validate_positive_integer(value, field_name):
        """
        Validates that the provided value is a positive integer.

        Parameters
        ----------
        value : str
            The value to validate.
        field_name : str
            The name of the field for error messaging.

        Returns
        -------
        bool
            True if the value is a positive integer, False otherwise.

        Notes
        -----
        If the value is not a positive integer, an error message is displayed.
        """
        if not value.isdigit() or int(value) < 0:
            sg.popup_error(f"Please enter a valid positive integer for {field_name}.")
            return False
        return True
