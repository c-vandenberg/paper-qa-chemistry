import os
from dotenv import load_dotenv
import tkinter as tk
from gui.paper_qa_gui import PaperQAGUI

load_dotenv()

ZOTERO_LIBRARY_ID: str = os.getenv('ZOTERO_USER_ID')
ZOTERO_API_KEY: str = os.getenv('ZOTERO_API_KEY')


def main():
    root = tk.Tk()
    app = PaperQAGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
