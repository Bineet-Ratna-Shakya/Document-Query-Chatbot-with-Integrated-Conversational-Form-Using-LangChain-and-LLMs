import os # can u see this commit in fact not? conmit hello
import os
import fitz  # PyMuPDF
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables from the specified .env file
env_path = '/Users/soul/Documents/chatbot-project/llama.env'
load_dotenv(dotenv_path=env_path)

# Initialize the tokenizer and model
model_name = "EleutherAI/gpt-neo-125M"  # Replace with the actual model name if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
llm = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to answer queries from documents
def answer_query(query, document_text):
    # Split the document into manageable chunks
    chunk_size = 2000
    chunks = [document_text[i:i + chunk_size] for i in range(0, len(document_text), chunk_size)]
    
    # Create the prompt
    answer = ""
    for chunk in chunks:
        prompt = f"Given the following document: {chunk}\n\nAnswer the following question: {query}"
        response = llm(prompt, max_new_tokens=200)[0]['generated_text']
        answer += response
    
    return answer

# Function to collect user information
def collect_user_info():
    with st.form(key='user_info_form'):
        name = st.text_input("Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email")
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            return name, phone, email
    return None, None, None

# Streamlit app
def main():
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

    # Collect user information
    if st.button("Call me"):
        name, phone, email = collect_user_info()
        if name and phone and email:
            st.write(f"User Info - Name: {name}, Phone: {phone}, Email: {email}")
        else:
            st.write("Please fill in all the fields.")

if __name__ == "__main__":
    main()
