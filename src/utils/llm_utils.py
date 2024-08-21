import tiktoken
import PyPDF2
import paperqa
import pickle
from pathlib import PosixPath
from typing import List


def extract_text_from_pdf(pdf_path: PosixPath) -> str:
    """
    Extracts text from a PDF file.

    Parameters
    ----------
    pdf_path : PosixPath
        The path to the PDF file from which text is to be extracted.

    Returns
    -------
    str
        The extracted text from all pages of the PDF.

    Notes
    -----
    This function uses the PyPDF2 library to read and extract text from each page of the PDF.
    The extracted text is concatenated and returned as a single string.
    """
    reader: PyPDF2.PdfReader = PyPDF2.PdfReader(pdf_path)
    text: str = ''
    for page in reader.pages:
        text += page.extract_text()
    return text


def calculate_tokens_from_pdf(pdf_path: PosixPath, model: str):
    """
    Calculates the number of tokens in a PDF document based on a specific language model.

    Tokens are chunks of texts that LLM models process. The token can be as short as one character or as long as one
    word. When a program interacts with an LLM API, both the input provided to the LLM, and the output provided by the
    LLM are tokenized.

    This provides a crude measure of how expensive each PDF will be, as the total number of tokens (input + output)
    determines the cost of that interaction.

    Parameters
    ----------
    pdf_path : PosixPath
        The path to the PDF file.
    model : str
        The name of the language model to be used for tokenization.

    Returns
    -------
    int
        The total number of tokens in the PDF document.

    Notes
    -----
    This function extracts text from the PDF and then encodes it using the specified language model's encoding.
    The token count is then returned as an integer.
    """
    pdf_text: str = extract_text_from_pdf(pdf_path)
    enc: tiktoken.Encoding = tiktoken.encoding_for_model(model)
    tokens: List = enc.encode(pdf_text)
    return len(tokens)
