from pydantic import BaseModel
from pathlib import Path


class ZoteroPaper(BaseModel):
    """
    A paper from Zotero.

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

    Notes
    -----
    This class is taken from the `zotero.py` module of the `paperqa` package, available at
    https://github.com/Future-House/paper-qa/blob/main/paperqa/contrib/zotero.py.
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
