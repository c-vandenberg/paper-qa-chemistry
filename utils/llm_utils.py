import tiktoken
import PyPDF2
import paperqa
import pickle
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


def load_paperqa_doc(pkl_file_path: str, llm_model: str) -> paperqa.Docs:
    try:
        with open('data/processed/supervised_learning_gpt_4o_mini.pkl', 'rb') as file:
            docs: paperqa.Docs = pickle.load(file)
            docs.llm = llm_model
            docs.set_client()
        print("Loaded previously pickled ChatGPT-4o Mini Docs object state")
    except FileNotFoundError:
        docs: paperqa.Docs = paperqa.Docs(llm=llm_model)
        print("No previously pickled ChatGPT-4o Mini Docs object state found. Starting fresh")

    return docs
