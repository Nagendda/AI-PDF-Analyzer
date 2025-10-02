# pdf_processor.py

import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file):
    """
    Extracts raw text from an uploaded PDF file.
    """
    text = ""
    try:
        # Open the PDF from the uploaded file's bytes
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
    except Exception as e:
        return f"Error reading PDF: {e}"

    # Basic cleaning
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_text_chunks(text):
    """
    Dear Students,
Good morning.

As per the schedule, you will be attending a 3-day Aptitude Training Program starting tomorrow.

You have received the pre-assessment test link from Smartica via your registered college email ID. Please check your inbox (and spam/junk folder, if necessary).

🕒 Test Window:
From 14.09.2025, 8:00 AM to 16.09.2025, 8:00 AM
You can take the test at any convenient time within this window.

✅ Important Notes:

The test is mandatory for all students.

We have shared the login procedure with you in both document and video format to assist you.

Make sure to complete the test within the given time frame.


Please ensure you attend the pre-assessment test without fail.
Best of luck!
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The size of each chunk in characters
        chunk_overlap=200   # The overlap between consecutive chunks
    )
    chunks = text_splitter.split_text(text)
    return chunks