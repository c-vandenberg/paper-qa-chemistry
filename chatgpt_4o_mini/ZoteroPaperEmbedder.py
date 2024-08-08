import os
import paperqa
import openai
import pickle
import time
from pydantic import BaseModel
from paperqa.contrib import ZoteroDB
from paperqa import utils as paperqa_utils
from pathlib import PosixPath, Path
from tqdm import tqdm
from typing import Generator, Union, Optional, List, Dict, cast
from pyzotero import zotero
from utils import llm_utils

ZOTERO_LIBRARY_ID: str = os.getenv('ZOTERO_USER_ID')

ZOTERO_API_KEY: str = os.getenv('ZOTERO_API_KEY')


class ZoteroPaper(BaseModel):
    """A paper from Zotero.

    Attributes:
    ----------
    key : str
        The citation key.
    title : str
        The title of the item.
    pdf : Path
        The path to the PDF for the item (pass to `paperqa.Docs`)
    num_pages : int
        The number of pages in the PDF.
    zotero_key : str
        The Zotero key for the item.
    details : dict
        The full item details from Zotero.
    """

    key: str
    title: str
    pdf: Path
    num_pages: int
    zotero_key: str
    details: dict

    def __str__(self) -> str:
        """Return the title of the paper."""
        return (
            f'ZoteroPaper(\n    key = "{self.key}",\n'
            f'title = "{self.title}",\n    pdf = "{self.pdf}",\n    '
            f'num_pages = {self.num_pages},\n    zotero_key = "{self.zotero_key}",\n    details = ...\n)'
        )


class ZoteroPaperEmbedder(ZoteroDB):
    def __init__(self, library_id, library_type, api_key):
        super().__init__(library_id=library_id, library_type=library_type, api_key=api_key)

    def chatgpt_4o_embedder(self, embedded_docs: paperqa.Docs, query_limit: int, query_start: int) -> paperqa.Docs:
        zotero: ZoteroDB = ZoteroDB(library_type='user')
        library_size: int = zotero.num_items()

        if query_start > library_size:
            print(f"Starting position ({query_start}) cannot be larger than Zotero database size ({library_size})")
            return embedded_docs

        papers: Generator = self.iterate(
            limit=query_limit,
            start=query_start,
            sort='title',
            direction='asc'
        )
        for i, paper in enumerate(tqdm(papers, desc="Processing Papers", ncols=100, miniters=1, mininterval=0.5),
                                  start=1):
            zotero_key = paper.details["key"]
            if zotero_key in embedded_docs.docnames:
                print(f"\nSkipping already processed paper {i}: {paper.title}")
                continue

            print(f"\nProcessing paper {i}: {paper.title}")

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
        """Given a search query, this will lazily iterate over papers in a Zotero library,
        downloading PDFs as needed."""

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
                    print(f"\nSkipping paper '{item['data']['title']}' as it has no associated PDF.")
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