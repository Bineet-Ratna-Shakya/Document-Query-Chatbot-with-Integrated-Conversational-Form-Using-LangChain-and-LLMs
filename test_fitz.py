import fitz  # PyMuPDF

# Sample PDF file path
sample_pdf_path = "Bineet_Ratna_Shakya_CV.pdf"  

try:
    # Open the PDF file
    doc = fitz.open(sample_pdf_path)
    print(f"Number of pages: {doc.page_count}")

    # Load the first page
    page = doc.load_page(0)
    print(f"Text on first page: {page.get_text()}")
except Exception as e:
    print(f"An error occurred: {e}")
