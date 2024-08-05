import os
import PyPDF2
import streamlit as st
from transformers import GPTJForCausalLM, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from the specified .env file
env_path = '/Users/soul/Documents/chatbot-project/openai.env'
load_dotenv(dotenv_path=env_path)

# Load the GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(model_name)

# Set the pad token as the eos token
tokenizer.pad_token = tokenizer.eos_token

# Initialize sentence transformer for embeddings
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, index, chunks, k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

def answer_query(query, document_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = [chunk for chunk in text_splitter.split_text(document_text)]

    index = create_faiss_index(chunks)
    retrieved_chunks = retrieve(query, index, chunks)
    context = " ".join([chunk for chunk, _ in retrieved_chunks])
    response = generate_response(f"Context: {context}\n\nQuestion: {query}")
    return response

def collect_user_info():
    with st.form(key='user_info_form'):
        name = st.text_input("Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email")
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            return name, phone, email
    return None, None, None

def main():
    st.title("Chatbot with Document Query and User Info Collection")

    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file is not None:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        document_text = extract_text_from_pdf("uploaded.pdf")
        st.write("Document uploaded successfully and text extracted.")

        user_query = st.text_input("Enter your query:")

        if st.button("Get Answer"):
            if user_query:
                answer = answer_query(user_query, document_text)
                st.write(f"Answer: {answer}")
            else:
                st.write("Please enter a query.")

    if st.button("Call me"):
        name, phone, email = collect_user_info()
        if name and phone and email:
            st.write(f"User Info - Name: {name}, Phone: {phone}, Email: {email}")
        else:
            st.write("Please fill in all the fields.")

if __name__ == "__main__":
    main()
