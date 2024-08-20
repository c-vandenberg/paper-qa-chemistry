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


def load_paperqa_doc(pkl_file_path: str, llm_model: str) -> paperqa.Docs:
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
        print("Loaded previously pickled ChatGPT-4o Mini Docs object state")
    except FileNotFoundError:
        docs: paperqa.Docs = paperqa.Docs(llm=llm_model, prompts=prompt_collection)
        print("No previously pickled ChatGPT-4o Mini Docs object state found. Starting fresh")

    return docs
