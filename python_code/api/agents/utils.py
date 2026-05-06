
# -*- coding: utf-8 -*-
import sys
import os
import subprocess
from dotenv import load_dotenv

try:
    # Try importing PyPDF2, install if needed (though we installed it)
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 library not found. PDF extraction will rely solely on pdftotext.")
    print("Consider installing it: pip install pypdf2")







def get_chatbot_response(client,model_name,messages,temperature=0.5):
    input_messages = []
    for message in messages:
        input_messages.append({"role": message["role"], "content": message["content"]})

    response = client.chat.completions.create(
        model=model_name,
        messages=input_messages,
        temperature=temperature,
        top_p=0.9,
        max_tokens=2000,
    ).choices[0].message.content
    
    return response

def get_embedding(embedding_client,model_name,text_input):
    output = embedding_client.embeddings.create(input = text_input,model=model_name)
    
    embedings = []
    for embedding_object in output.data:
        embedings.append(embedding_object.embedding)

    return embedings

def double_check_json_output(client,model_name,json_string):
    prompt = f""" You will check this json string and correct any mistakes that will make it invalid. Then you will return the corrected json string. Nothing else. 
    If the Json is correct just return it.

    Do NOT return a single letter outside of the json string.

    {json_string}
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_chatbot_response(client,model_name,messages)

    return response












def read_text_file(file_path):
    """Reads content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return None

def _extract_text_with_pdftotext(pdf_path):
    """Internal function to attempt extraction using pdftotext."""
    try:
        command = ["pdftotext", "-enc", "UTF-8", pdf_path, "-"]
        print(f"Attempting extraction with command: {' '.join(command)}")

        result = subprocess.run(command, capture_output=True, check=True, timeout=60)
        stdout_bytes = result.stdout
        try:
            print("Decoding pdftotext output as UTF-8...")
            return stdout_bytes.decode("utf-8")
        except UnicodeDecodeError:
            print("UTF-8 decoding failed. Trying latin-1...")
            try:
                return stdout_bytes.decode("latin-1")
            except UnicodeDecodeError:
                print("latin-1 decoding failed. Decoding UTF-8 with replacement...")
                return stdout_bytes.decode("utf-8", errors="replace")
    except FileNotFoundError:
        print("pdftotext command not found or PDF path incorrect. Ensure poppler-utils is installed and in PATH.")
        return None # Indicate pdftotext failure
    except subprocess.TimeoutExpired:
        print(f"pdftotext command timed out for {pdf_path}.")
        return None # Indicate pdftotext failure
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode("utf-8", errors="replace") if e.stderr else "(No stderr)"
        print(f"pdftotext failed for {pdf_path}. Error: {e}. Stderr: {stderr_output}")
        return None # Indicate pdftotext failure
    except Exception as e:
        print(f"Unexpected error during pdftotext extraction for {pdf_path}: {e}")
        return None # Indicate pdftotext failure

def _extract_text_with_pypdf2(pdf_path):
    """Internal function to attempt extraction using PyPDF2 library."""
    if not PYPDF2_AVAILABLE:
        print("PyPDF2 library is not available. Cannot use PyPDF2 extraction.")
        return None
        
    print(f"Attempting extraction with PyPDF2 for: {pdf_path}")
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"PyPDF2 found {num_pages} pages.")
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text: # Check if text extraction returned something
                    text += page_text + "\n" # Add newline between pages
                else:
                    print(f"Warning: PyPDF2 extracted no text from page {page_num + 1}.")
        print("PyPDF2 extraction completed.")
        return text if text else None # Return None if no text was extracted at all
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path} for PyPDF2.")
        return None
    except Exception as e:
        # Catch potential PyPDF2 specific errors (e.g., encrypted PDF)
        print(f"Error extracting text with PyPDF2 from {pdf_path}: {e}")
        if "encrypted" in str(e).lower():
             print("PDF might be password protected.")
        return None

def extract_text_from_pdf(pdf_path, method="auto"):
    """Extracts text from PDF using pdftotext (preferred) or PyPDF2 (fallback).

    Args:
        pdf_path (str): Path to the PDF file.
        method (str): Extraction method ("auto", "pdftotext", "pypdf2"). 
                      "auto" tries pdftotext first, then PyPDF2 on failure.

    Returns:
        str or None: Extracted text, or None if extraction fails.
    """
    extracted_text = None

    if method in ["auto", "pdftotext"]:
        extracted_text = _extract_text_with_pdftotext(pdf_path)
        if extracted_text is not None:
            print("Successfully extracted text using pdftotext.")
            return extracted_text
        elif method == "pdftotext":
            print("pdftotext method failed as requested. No fallback attempted.")
            return None
        else: # method == "auto" and pdftotext failed
            print("pdftotext failed. Attempting fallback with PyPDF2...")

    if method in ["auto", "pypdf2"]:
        extracted_text = _extract_text_with_pypdf2(pdf_path)
        if extracted_text is not None:
            print("Successfully extracted text using PyPDF2.")
            return extracted_text
        else:
            print("PyPDF2 method also failed or returned no text.")
            return None
            
    print(f"Error: Invalid extraction method specified: {method}")
    return None




