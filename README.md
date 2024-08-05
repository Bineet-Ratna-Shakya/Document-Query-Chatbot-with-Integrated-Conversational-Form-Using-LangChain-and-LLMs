# Document-Query-Chatbot-with-Integrated-Conversational-Form-Using-LangChain-and-LLMs

# Chatbot Project

## Project Description
This chatbot is designed to answer user queries from various documents and includes a conversational form for collecting user information such as Name, Phone Number, and Email. It integrates with APIs such as OpenAI, GPT-J, and LLaMA to provide intelligent and responsive interactions.

## Features
- **API Integration**: Utilizes OpenAI GPT, GPT-J (running locally), and LLaMA for natural language understanding and responses.
- **Document Query Processing**: Answers user queries based on the content of provided documents.
- **Form**: Collects user information.


### Prerequisites
- Python 3.11.4
- Required libraries (specified in `requirements.txt`)

Warning
This project was developed on macOS. Please ensure compatibility with your operating system. Make sure to install Python 3.11.4 and all required libraries.

### Steps to Set Up the Project
1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/chatbot-project.git
    ```
2. **Navigate to the project directory**:
    ```sh
    cd chatbot-project
    ```
3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
    
### Creating a Virtual Environment

If you don't have a virtual environment set up, follow these steps to create one:

1. **Install `virtualenv` (if not already installed):**
   ```bash
   pip install virtualenv
   
2. **Activate the virtual environment:**

    - **On Windows:**

        ```bash
        .\venv\Scripts\activate
        ```

    - **On macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

## Installing the Required Libraries

Once the virtual environment is activated, install the necessary libraries:

```bash
pip install torch transformers sentence-transformers faiss-cpu streamlit
```
Note: If you are using a GPU, ensure you install the GPU-compatible versions of the libraries.

## Additional Setup

### CUDA and cuDNN
To leverage GPU acceleration, ensure CUDA and cuDNN are installed and configured correctly. Follow the official installation guides for your specific GPU and operating system.

### Verifying the Setup
You can verify your setup by running a small script to check if the libraries are installed correctly and the GPU is being utilized.

```python
import torch
print(torch.cuda.is_available())
```

### Configuration Instructions
1. **API Keys and Environment Variables**:
   - Create a `.env` file in the root directory and add your API keys and necessary environment variables:
     ```env
     LLAMA_API_KEY=your_llama_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage

### Basic Commands and Interactions
1. **Run the chatbot**:
    ```sh
    streamlit run /your/path/directly/chatbot-project/OPENAILLM.py   # Uses OpenAI GPT LLM
    streamlit run /your/path/directly/chatbot-project/GPT-JModel.py  # Uses GPT-J running locally
    streamlit run /your/path/directly/chatbot-project/llamaModel.py  # Uses LLaMA
    ```
2. **Interacting with the chatbot**:
    - Type your queries or commands into the chatbot interface.
    - For document-based queries, ensure your documents are placed in the correct directory and are in the supported format.
3. **Triggering the conversational form**:
    - The chatbot will automatically prompt for user information when required.

### Example Usage
1. **Querying Documents**:
    - User: "Can you summarize the main points of this document?"
    - Chatbot: "Sure, here are the main points: [summary based on document content]"
2. **Providing User Information**: (future development).
    - User: "I'd like to provide my contact information."
    - Chatbot: "Please provide your name, phone number, and email address."

### Supported Document Formats
- PDF

  
## API Integration

### Setting Up and Configuring APIs
1. **LangChain**:
   - Follow the [LangChain documentation]((https://python.langchain.com/v0.2/docs/introduction/)) to set up.
2. **LLAMA**:
   - Refer to the [LLAMA API documentation]((https://docs.llama-api.com/quickstart)) for setup instructions.
3. **OpenAI**:
   - Visit the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/introduction) for configuration details.


# Resources Needed for Running GPT-J (6 Billion Parameters)

Running a large model like GPT-J (6 billion parameters) requires significant computational resources. Below is a general overview of the necessary resources.

## Memory (RAM)
- **GPU Memory (VRAM):** At least 16 GB of GPU memory is recommended.
- **System RAM:** At least 32 GB of system RAM is recommended.

## Computational Power
- **GPU:** A high-performance GPU such as NVIDIA RTX 3090, A100, or V100 is ideal.
- **CPU:** Modern processors with multiple cores (e.g., Intel i7/i9, AMD Ryzen 7/9) are recommended.

## Storage
- **Disk Space:** At least 50-100 GB of free space.

## Network Bandwidth
- **Internet Connection:** A stable connection with sufficient bandwidth is necessary.

## Software Requirements
- **CUDA:** For GPU acceleration, CUDA and cuDNN must be installed.
- **Libraries:** Install compatible versions of the following libraries:
  - PyTorch
  - Transformers
  - Sentence Transformers
  - FAISS
  - Streamlit


## Stability and Performance
- **OpenAI GPT**: Most stable and reliable for generating accurate responses.
- **GPT-J**: Requires significant computational resources but performs well for generating responses.
- **LLaMA**: Suitable for extracting text but may struggle with accurately responding to queries.

  
## Libraries Used
- **Transformers Library**: Using `GPTJForCausalLM`, `GPT2Tokenizer`, `pipeline`, `AutoTokenizer`, and `AutoModelForCausalLM` from the Hugging Face Transformers library for natural language processing tasks.
- **Sentence Transformers**: Using `SentenceTransformer` for embedding sentences.
- **FAISS**: Using Facebook AI Similarity Search (FAISS) for efficient similarity search and clustering of dense vectors.
- **LangChain**: Using `LLMChain`, `PromptTemplate`, `OpenAI`, `CharacterTextSplitter`, and `RecursiveCharacterTextSplitter` for building language model chains and processing text.
- **PyPDF2** and **PyMuPDF (fitz)**: For handling and extracting text from PDF files.
- **Streamlit**: For creating the web interface of the chatbot.
- **dotenv**: For loading environment variables from a `.env` file.
  

### Conversational Form Implementation
- **Triggering the Form**: The form is triggered by clicking on the "Call Me" button.
- **Data Handling**: Collected user data is stored securely in accordance with privacy guidelines.( Further Development)

## Further Development
- **Call Me Function**: The current implementation displays a form to collect user information. In future developments, this feature will be enhanced to support chat functionality. Here are the key points for the enhancement:

  1. **Real-Time Communication**: Enable real-time chat interactions between the user and the bot to provide immediate feedback and support.
  
  2. **User Authentication**: Implement user authentication to ensure secure and personalized interactions. This may include login functionality and user session management.
  
  3. **Enhanced Data Collection**: Allow the bot to dynamically ask follow-up questions based on user responses, making the data collection process more interactive and thorough.
  
  4. **Context Management**: Develop the bot's ability to maintain context over multiple turns in the conversation, allowing it to refer back to previous interactions and provide more coherent responses.
  
  5. **Integration with Communication Channels**: Extend the bot's capabilities to integrate with various communication platforms such as WhatsApp, Messenger, or custom web chat interfaces for broader accessibility.
  
  6. **Natural Language Understanding Improvements**: Continuously improve the bot's natural language understanding to accurately interpret user queries and provide relevant responses.
  
  7. **Personalized Responses**: Use collected user information to tailor responses and interactions, creating a more personalized and engaging user experience.
  
  8. **Feedback Loop**: Implement a feedback loop where users can rate the quality of responses, helping to fine-tune the bot's performance and improve future interactions.
  
  9. **Automated Follow-Ups**: Enable the bot to schedule and send automated follow-up messages or emails based on the user information collected, ensuring ongoing engagement.
  
  10. **Scalability**: Ensure that the chat functionality can handle multiple concurrent users without degradation in performance, making the bot suitable for large-scale deployments.
  
  11. **Compliance and Privacy**: Adhere to data protection regulations and privacy standards, ensuring that all collected user information is stored and processed securely.
  
  12. **Analytics and Reporting**: Develop analytics tools to monitor user interactions, collect insights, and generate reports to track the bot's performance and user satisfaction.

By implementing these enhancements, the "Call Me" function will transform into a comprehensive chat feature that not only collects user information but also engages users in meaningful and interactive conversations.


### Guidelines for Contributing
- **Reporting Issues**: Use the issue tracker on GitHub to report bugs or suggest features.
- **Submitting Contributions**: Fork the repository, create a new branch, and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact Information
For further questions or support, contact:
- **Email**: bineetshakyaa@gmail.com
- **GitHub**: 

## Acknowledgements
This project leverages the following open-source libraries and tools:
- [OpenAI](https://openai.com/)
- [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)
- [Transformers](https://huggingface.co/transformers/)
- [Sentence Transformers](https://www.sbert.net/docs/)
- [FAISS](https://faiss.ai/)
- [LangChain](https://langchain.com/)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)
- [Streamlit](https://streamlit.io/)

## Changelog
- **v1.0.0**: Initial release with basic features for document query processing and user information collection.
