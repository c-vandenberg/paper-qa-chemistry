import tiktoken
import PyPDF2
from pathlib import PosixPath
from typing import List


def extract_text_from_pdf(pdf_path: PosixPath) -> str:
    reader: PyPDF2.PdfReader = PyPDF2.PdfReader(pdf_path)
    text: str = ''
    for page in reader.pages:
        text += page.extract_text()
    return text


def calculate_tokens_from_pdf(pdf_path: PosixPath, model: str):
    pdf_text: str = extract_text_from_pdf(pdf_path)
    enc: tiktoken.Encoding = tiktoken.encoding_for_model(model)
    tokens: List = enc.encode(pdf_text)
    return len(tokens)
