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
    prompts = (
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
    prompt_collection = paperqa.PromptCollection(qa=prompts)

    try:
        with open(pkl_file_path, 'rb') as file:
            docs: paperqa.Docs = pickle.load(file)
            docs.llm = llm_model
            docs.prompts = None
            docs.prompts = prompt_collection
            docs.set_client()
        print("Loaded previously pickled ChatGPT-4o Mini Docs object state")
    except FileNotFoundError:
        docs: paperqa.Docs = paperqa.Docs(llm=llm_model, prompts=prompt_collection)
        print("No previously pickled ChatGPT-4o Mini Docs object state found. Starting fresh")

    return docs
