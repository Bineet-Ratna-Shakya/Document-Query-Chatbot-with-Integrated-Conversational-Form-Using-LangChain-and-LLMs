# Document-Query-Chatbot-with-Integrated-Conversational-Form-Using-LangChain-and-LLMs

# Chatbot Project

## Project Description
This chatbot is designed to answer user queries from various documents and includes a conversational form for collecting user information such as Name, Phone Number, and Email. It integrates with APIs such as OpenAI, GPT-J, and LLaMA to provide intelligent and responsive interactions.

## Features
- **API Integration**: Utilizes OpenAI GPT, GPT-J (running locally), and LLaMA for natural language understanding and responses.
- **Document Query Processing**: Answers user queries based on the content of provided documents.
- **Form**: Collects user information.


### Prerequisites
- Python 3.x
- Required libraries (specified in `requirements.txt`)

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

## API Integration

### Setting Up and Configuring APIs
1. **LangChain**:
   - Follow the [LangChain documentation](https://langchain.com/docs) to set up your API key and integrate it into the project.
2. **Gemini**:
   - Refer to the [Gemini API documentation](https://gemini.com/api) for setup instructions.
3. **OpenAI**:
   - Visit the [OpenAI API documentation](https://openai.com/api) for configuration details.


Running a large model like GPT-J (6 billion parameters) requires significant computational resources. Here’s a general overview of the resources needed:
1. Memory (RAM)
* GPU Memory (VRAM): At least 16 GB of GPU memory is recommended. GPT-J can be run on GPUs with less VRAM, but performance will be slower, and you might need to use techniques like model parallelism or offloading to CPU, which can reduce efficiency.
* System RAM: At least 32 GB of system RAM is recommended. Large models and their associated data (like embeddings and intermediate computations) can consume substantial system memory.
2. Computational Power
* GPU: A high-performance GPU such as NVIDIA RTX 3090, A100, or V100 is ideal. The GPU should have sufficient memory to handle the model’s parameters and intermediate computations.
* CPU: A multi-core CPU can also help, especially for preprocessing and managing data. Modern processors with multiple cores (e.g., Intel i7/i9, AMD Ryzen 7/9) are recommended.
3. Storage
* Disk Space: Ensure you have sufficient disk space for storing the model weights and any temporary data generated during computation. For GPT-J, the model weights alone are around 25 GB. Ensure you have at least 50-100 GB of free space to accommodate the model and related files.
4. Network Bandwidth
* Internet Connection: Downloading the model and dependencies requires a good internet connection. A stable connection with sufficient bandwidth is necessary to handle large files efficiently.
5. Software Requirements
* CUDA: For GPU acceleration, CUDA and cuDNN must be installed. Compatibility depends on the GPU and the version of PyTorch used.
* Libraries: Make sure to install compatible versions of libraries like PyTorch, Transformers, Sentence Transformers, FAISS, and Streamlit.
Alternative Approaches
1. Model Optimization:
    * Distillation: Use a smaller, distilled version of the model if possible. Distilled models are lighter and faster while maintaining reasonable performance.
    * Quantization: This technique reduces the precision of the model weights to decrease memory usage and speed up inference.
2. Cloud Solutions:
    * Managed Services: Consider using cloud-based machine learning services (like AWS Sagemaker, Google AI Platform, or Azure Machine Learning) that provide powerful GPUs and manage the infrastructure for you.
    * Pre-built Models: Use hosted APIs for models like OpenAI’s GPT-3 or GPT-4, which offload the computational burden to their infrastructure.
3. Model Parallelism:
    * Sharding: For very large models, distribute the model across multiple GPUs or machines to handle larger models that exceed the memory capacity of a single device.
Testing Resource Requirements
Before deploying the full model, you can start by testing with smaller models or subsets of data to gauge the resource requirements and ensure your environment is appropriately configured.
By ensuring you meet these resource requirements, you can effectively run large models like GPT-J and avoid performance bottlenecks.

  
## Libraries Used
- **Transformers Library**: Using `GPTJForCausalLM`, `GPT2Tokenizer`, `pipeline`, `AutoTokenizer`, and `AutoModelForCausalLM` from the Hugging Face Transformers library for natural language processing tasks.
- **Sentence Transformers**: Using `SentenceTransformer` for embedding sentences.
- **FAISS**: Using Facebook AI Similarity Search (FAISS) for efficient similarity search and clustering of dense vectors.
- **LangChain**: Using `LLMChain`, `PromptTemplate`, `OpenAI`, `CharacterTextSplitter`, and `RecursiveCharacterTextSplitter` for building language model chains and processing text.
- **PyPDF2** and **PyMuPDF (fitz)**: For handling and extracting text from PDF files.
- **Streamlit**: For creating the web interface of the chatbot.
- **dotenv**: For loading environment variables from a `.env` file.
  

### Conversational Form Implementation
- **Triggering the Form**: The form is automatically triggered by specific keywords or commands during the conversation.
- **Data Handling**: Collected user data is stored securely in accordance with privacy guidelines. 


### Guidelines for Contributing
- **Reporting Issues**: Use the issue tracker on GitHub to report bugs or suggest features.
- **Submitting Contributions**: Fork the repository, create a new branch, and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact Information
For further questions or support, contact:
- **Email**: your.email@example.com
- **GitHub**: [yourusername](https://github.com/yourusername)

## Acknowledgements
This project utilizes the following third-party libraries and tools:
- [LangChain](https://langchain.com/)
- [Gemini](https://gemini.com/)
- [OpenAI](https://openai.com/)

## Changelog
- **v1.0.0**: Initial release with basic features for document query processing and user information collection.
