## RAG Chatbot Llama
This is a RAG chatbot that uses: 
- Llama 3.1 (8B)
- Ollama
- LangChain
- ChromaDB vector database
- GIST vector embeddings (small)
- FastAPI

Here is an explanation of what important files do:

    ├── chatbot_api  
    │   ├── scraping.py                               : scraps some static web pages, stores it inside chatbot_api/documents as .txt file  
    │   ├── sentence_transformer_langchain_wrapper.py : a wrapper for SentenceTransformers to make it compatible with LangChain's Embeddings interface
    │   ├── chromadb_storage.py                       : takes all .txt file in chatbot_api/documents, divide it into chunks, get its vector embeddings, then store it into /chroma_db  
    │   ├── retrieval.py                              : takes in a text query and retrieves relevant documents in the vector database  
    │   ├── chatbot.py                                : takes in a text query, finds relevant documents using retrieval.py, augments the query with the documents, then queries Llama 3.1
    │   ├── main.py                                   : FastAPI API app
    │  
    ├── frontend (todo)