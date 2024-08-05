import os
import fitz  # PyMuPDF
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# Loading environment variables from the specified .env file
env_path = '/Users/soul/Documents/chatbot-project/openai.env'
load_dotenv(dotenv_path=env_path)

# Setting up OpenAI API key from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    st.error("Did not find OPENAI_API_KEY. Please check your .env file.")
    st.stop()

# Initializing the LLM with OpenAI
llm = OpenAI(api_key=openai_api_key)

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def answer_query(query, document_text):
    """
    Answers a query based on the content of a document.

    Args:
        query (str): The query to be answered.
        document_text (str): The text extracted from the document.

    Returns:
        str: The answer to the query.
    """
    # Splitting the document into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.split_text(document_text)
    
    # Creating the prompt template
    prompt_template = PromptTemplate(
        template="Given the following document: {document}\n\nAnswer the following question: {question}",
        input_variables=["document", "question"]
    )

    # Initializing the LLM chain
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )
    
    # Iterating over chunks to get the answer
    answer = ""
    for chunk in chunks:
        response = llm_chain.run({"document": chunk, "question": query})
        answer += response
    
    return answer

def collect_user_info():
    """
    Collects user information through a form.

    Returns:
        tuple: A tuple containing the name, phone number, and email of the user.
    """
    with st.form(key='user_info_form'):
        name = st.text_input("Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email")
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            return name, phone, email
    return None, None, None

def main():
    """
    Main function to run the Streamlit app for document query and user info collection.
    """
    st.title("Chatbot with Document Query and User Info Collection")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file is not None:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        document_text = extract_text_from_pdf("uploaded.pdf")
        st.write("Document uploaded successfully and text extracted.")

        # User query input
        user_query = st.text_input("Enter your query:")

        if st.button("Get Answer"):
            if user_query:
                answer = answer_query(user_query, document_text)
                st.write(f"Answer: {answer}")
            else:
                st.write("Please enter a query.")

    # Collecting user information
    if st.button("Call me"):
        name, phone, email = collect_user_info()
        if name and phone and email:
            st.write(f"User Info - Name: {name}, Phone: {phone}, Email: {email}")
        else:
            st.write("Please fill in all the fields.")

if __name__ == "__main__":
    main()
