import os
import sys
import paperqa
import openai
import pickle
import time
import PySimpleGUI as sg
from paperqa.contrib import ZoteroDB
from paperqa import utils as paperqa_utils
from pathlib import PosixPath, Path
from tqdm import tqdm
from typing import Generator, Optional, List, cast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.zotero_paper import ZoteroPaper
from utils import llm_utils

ZOTERO_LIBRARY_ID: str = os.getenv('ZOTERO_USER_ID')

ZOTERO_API_KEY: str = os.getenv('ZOTERO_API_KEY')


class ZoteroPaperEmbedder(ZoteroDB):
    """
    A class for embedding papers from a Zotero database into a set of vectors using a specified language model.

    This class extends the `ZoteroDB` class from the `paperqa` package and provides additional functionality
    for embedding papers from Zotero into a `paperqa.Docs` object. It also includes methods for outputting
    logs to a PySimpleGUI interface.

    Attributes
    ----------
    console_multiline : sg.Multiline
        A PySimpleGUI Multiline element for logging output.
    window : sg.Window
        The PySimpleGUI window that contains the Multiline element.

    Methods
    -------
    console_output(message: str)
        Logs a message to the Multiline element or prints it to the console.
    load_paperqa_doc(pkl_file_path: str, llm_model: str) -> paperqa.Docs
        Loads a paperqa.Docs object from a pickle file, or creates a new one if the file does not exist.
    chatgpt_4o_embedder(embedded_docs: paperqa.Docs, query_limit: int, query_start: int) -> paperqa.Docs
        Embeds papers from Zotero into the given `paperqa.Docs` object.
    iterate(limit: int = 25, start: int = 0, q: Optional[str] = None, qmode: Optional[str] = None,
            since: Optional[str] = None, tag: Optional[str] = None, sort: Optional[str] = None,
            direction: Optional[str] = None,
            collection_name: Optional[str] = None) -> Generator[ZoteroPaper, None, None]
        Lazily iterates over papers in a Zotero library and downloads PDFs as needed.
    _get_citation_key(item: dict) -> str
        Generates a citation key for a Zotero item based on its metadata.

    Notes
    -----
    This class extends and overrides the `ZoteroDB` class implementation found in the `zotero.py` module of the
    `paperqa` package, available at https://github.com/Future-House/paper-qa/blob/main/paperqa/contrib/zotero.py.
    """
    def __init__(self, library_id: Optional[str], library_type: str, api_key: Optional[str], console_multiline=None,
                 window=None):
        super().__init__(library_id=library_id, library_type=library_type, api_key=api_key)
        self.console_multiline = console_multiline
        self.window = window

    def console_output(self, message: str):
        """
        Logs a message to the Multiline element or prints it to the console.

        Parameters
        ----------
        message : str
            The message to log.

        Notes
        -----
        If `console_multiline` is provided, the message is logged to the PySimpleGUI Multiline element.
        Otherwise, it is printed to the console.
        """
        if self.console_output:
            self.console_multiline.print(message)
            self.window.refresh()
        else:
            print(message)

    def load_paperqa_doc(self, pkl_file_path: str, llm_model: str) -> paperqa.Docs:
        """
        Loads a paperqa.Docs object from a pickle file, or creates a new one if the file does not exist.

        Parameters
        ----------
        pkl_file_path : str
            The path to the pickle file containing the saved Docs object.
        llm_model : str
            The language model to be used for the Docs object.

        Returns
        -------
        paperqa.Docs
            The loaded or newly created Docs object.

        Notes
        -----
        The function first attempts to load a Docs object from the specified pickle file.
        If the file does not exist, a new Docs object is created with the specified language model and prompts.
        The Docs object is configured to use a set of predefined prompts for answering questions, and its client is set up.
        """
        prompts: str = (
            "Answer the question '{question}' "
            "Use the context below if helpful. "
            "Context: {context}\n\n"
            "Include all relevant relevant academic papers and documents in your answer. "
            "If the information is insufficient or ambiguous, provide a brief explanation of what additional "
            "information would be necessary to fully answer the question. "
            "Evaluate the relevance of each source in relation to the question. "
            "Do not include a separate references section within the body of the text. "
            "Use Harvard style for references when referencing sources\n\n"
        )
        prompt_collection: paperqa.PromptCollection = paperqa.PromptCollection(qa=prompts)

        try:
            with open(pkl_file_path, 'rb') as file:
                docs: paperqa.Docs = pickle.load(file)
                docs.llm = llm_model
                docs.prompts = None
                docs.prompts = prompt_collection
                docs.set_client()
            self.console_output("Loaded previously pickled `Docs` object state")
        except FileNotFoundError:
            docs: paperqa.Docs = paperqa.Docs(llm=llm_model, prompts=prompt_collection)
            self.console_output("No previously pickled `Docs` object state found. Starting fresh")

        return docs

    def embed_docs(self, embedded_docs: paperqa.Docs, query_limit: int, query_start: int) -> paperqa.Docs:
        """
        Embeds papers from the Zotero database into vectors within a `paperqa.Docs` object.

        Parameters
        ----------
        embedded_docs : paperqa.Docs
            The document set to which papers will be embedded as vectors.
        query_limit : int
            The number of papers to embed into the document set.
        query_start : int
            The starting position in the Zotero database to begin the embedding.

        Returns
        -------
        paperqa.Docs
            The updated document set with the newly embedded paper vectors.

        Notes
        -----
        This method processes papers from the Zotero database, checking for duplicates and handling potential
        errors such as API rate limits.
        """
        zotero: ZoteroDB = ZoteroDB(library_type='user')
        library_size: int = zotero.num_items()
        llm_model: str = embedded_docs.llm
        pkl_file_path: str = (f"../data/processed/paper_qa_"
                              f"{llm_model.lower().replace(' ', '_').replace('-', '_')}.pkl")

        if query_start > library_size:
            sg.popup_error(f"Starting position ({query_start}) cannot be larger than Zotero database size "
                           f"({library_size})")
            return embedded_docs

        papers: Generator = self.iterate(
            limit=query_limit,
            start=query_start,
            sort='dateAdded',
            direction='desc'
        )

        for i, paper in enumerate(tqdm(papers, desc="Processing Papers", ncols=100, miniters=1, mininterval=0.5),
                                  start=1):
            zotero_key = paper.details["key"]
            if zotero_key in embedded_docs.docnames:
                self.console_output(f"\nSkipping already processed paper {i}: {paper.title}")
                continue

            self.console_output(f"\nProcessing paper {i}: {paper.title}")

            paper_content: PosixPath = paper.pdf
            num_tokens: int = llm_utils.calculate_tokens_from_pdf(paper_content, 'gpt-4o-mini')

            self.console_output(f"\nPaper contains {num_tokens} input tokens")

            try:
                embedded_docs.add(paper.pdf, docname=zotero_key)
            except openai.RateLimitError as e:
                sg.popup_error(f"\nRate limit exceeded: {e}. Waiting 60s before retrying...")
                time.sleep(60)  # Wait for 60 seconds before retrying
                continue
            except openai.OpenAIError as e:
                sg.popup_error(f"\nOpenAI API error: {e}")
                break
            except Exception as e:
                sg.popup_error(f"\nUnexpected error: {e}")
                break

            with open(pkl_file_path, 'wb') as file:
                pickle.dump(embedded_docs, file)
            self.console_output(f"\nSaved checkpoint after processing paper {i}.")

    def iterate(
            self,
            limit: int = 25,
            start: int = 0,
            q: Optional[str] = None,
            qmode: Optional[str] = None,
            since: Optional[str] = None,
            tag: Optional[str] = None,
            sort: Optional[str] = None,
            direction: Optional[str] = None,
            collection_name: Optional[str] = None,
    ):
        """Given a search query, this will lazily iterate over papers in a Zotero library, downloading PDFs as needed.

        This will download all PDFs in the query.
        For information on parameters, see
        https://pyzotero.readthedocs.io/en/latest/?badge=latest#zotero.Zotero.add_parameters
        For extra information on the query, see
        https://www.zotero.org/support/dev/web_api/v3/basics#search_syntax.

        For each item, it will return a `ZoteroPaper` object, which has the following fields:

            - `pdf`: The path to the PDF for the item (pass to `paperqa.Docs`)
            - `key`: The citation key.
            - `title`: The title of the item.
            - `details`: The full item details from Zotero.

        Parameters
        ----------
        q : str, optional
            Quick search query. Searches only titles and creator fields by default.
            Control with `qmode`.
        qmode : str, optional
            Quick search mode. One of `titleCreatorYear` or `everything`.
        since : int, optional
            Only return objects modified after the specified library version.
        tag : str, optional
            Tag search. Can use `AND` or `OR` to combine tags.
        sort : str, optional
            The name of the field to sort by. One of dateAdded, dateModified,
            title, creator, itemType, date, publisher, publicationTitle,
            journalAbbreviation, language, accessDate, libraryCatalog, callNumber,
            rights, addedBy, numItems (tags).
        direction : str, optional
            asc or desc.
        limit : int, optional
            The maximum number of items to return. Default is 25. You may use the `start`
            parameter to continue where you left off.
        start : int, optional
            The index of the first item to return. Default is 0.

        Yields
        ------
        ZoteroPaper
            An instance of `ZoteroPaper` for each paper retrieved.

        Notes
        -----
        This method overrides the `ZoteroDB.iterate()` class method found in the `zotero.py` module of the `paperqa`
        package, available at https://github.com/Future-House/paper-qa/blob/main/paperqa/contrib/zotero.py.

        Overriding was necessary as there was a bug within the original method logic that prevented embedding past the
        first 100 papers in the Zotero database.
        """
        query_kwargs = {}

        if q is not None:
            query_kwargs["q"] = q
        if qmode is not None:
            query_kwargs["qmode"] = qmode
        if since is not None:
            query_kwargs["since"] = since
        if tag is not None:
            query_kwargs["tag"] = tag
        if sort is not None:
            query_kwargs["sort"] = sort
        if direction is not None:
            query_kwargs["direction"] = direction

        if collection_name is not None and len(query_kwargs) > 0:
            raise ValueError(
                "You cannot specify a `collection_name` and search query simultaneously!"
            )

        max_limit = 100
        items: List = []
        pdfs: List[Path] = []

        collection_id = None
        if collection_name:
            collection_id = self._get_collection_id(collection_name)

        num_remaining = limit

        while num_remaining > 0:
            cur_limit = min(max_limit, num_remaining)
            self.logger.info(f"Downloading new batch of up to {cur_limit} papers.")

            if collection_id:
                _items = self._sliced_collection_items(
                    collection_id, limit=cur_limit, start=start
                )
            else:
                _items = self.top(**query_kwargs, limit=cur_limit, start=start)

            if len(_items) == 0:
                break

            self.logger.info("Downloading PDFs.")
            _pdfs = [self.get_pdf(item) for item in _items]

            # Filter:
            for item, pdf in zip(_items, _pdfs):
                no_pdf = item is None or pdf is None
                is_duplicate = pdf in pdfs

                if no_pdf or is_duplicate:
                    self.console_output(f"\nSkipping paper '{item['data']['title']}' as it has no associated PDF.")
                    continue
                pdf = cast(Path, pdf)
                title = item["data"].get("title", "")
                yield ZoteroPaper(
                    key=self._get_citation_key(item),
                    title=title,
                    pdf=pdf,
                    num_pages=paperqa_utils.count_pdf_pages(pdf),
                    details=item,
                    zotero_key=item["key"],
                )
                items.append(item)
                pdfs.append(pdf)

            start += cur_limit  # Increment start by the number of items processed
            num_remaining -= cur_limit  # Decrease remaining items to process

        self.logger.info("Finished downloading papers. Now creating Docs object.")

    @staticmethod
    def _get_citation_key(item: dict) -> str:
        """
        Generates a citation key for a Zotero item based on its metadata.

        Parameters
        ----------
        item : dict
            The metadata of the Zotero item.

        Returns
        -------
        str
            The generated citation key, which includes the author's last name, a short title, and the date.

        Notes
        -----
        This method uses the `ZoteroDB` class implementation found in the `zotero.py` module of the
        `paperqa` package, available at https://github.com/Future-House/paper-qa/blob/main/paperqa/contrib/zotero.py.

        No changes have been made to the method logic.
        """
        if (
                "data" not in item
                or "creators" not in item["data"]
                or len(item["data"]["creators"]) == 0
                or "lastName" not in item["data"]["creators"][0]
                or "title" not in item["data"]
                or "date" not in item["data"]
        ):
            return item["key"]

        last_name = item["data"]["creators"][0]["lastName"]
        short_title = "".join(item["data"]["title"].split(" ")[:3])
        date = item["data"]["date"]

        # Delete non-alphanumeric characters:
        short_title = "".join([c for c in short_title if c.isalnum()])
        last_name = "".join([c for c in last_name if c.isalnum()])
        date = "".join([c for c in date if c.isalnum()])

        return f"{last_name}_{short_title}_{date}_{item['key']}".replace(" ", "")
