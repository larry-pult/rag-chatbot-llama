from langchain_ollama import ChatOllama
from langchain_core.prompts.chat import ChatPromptTemplate
import retrieval

MODEL = ChatOllama(
    model="llama3.1:8b", 
    base_url="http://127.0.0.1:11434"
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
"""You are a retrieval augmented generation chatbot that answers a question based on documents obtained from an information retrieval system.
Each document retrieved will have a text content, a source, and a confidence score ranging from 0 to 1 (the higher the score is, the more likely the corresponding document is related to the question).
However, the retrieval system is not perfect. If a retrieved document is irrelevant with the question, or contains contradictory information, please do not include the information from that document in the answer.
If none of the retrieved documents are relevant to the question, you should say that you don't know the answer.
Please provide the answer in your own words, paraphrasing and expanding on the relevant documents, but please do not include the raw document itself nor any of its metadata.

Retrieved Documents: {retrieved_documents}

Question: {question}

Answer:"""
)

QUERY_CHAIN = PROMPT_TEMPLATE | MODEL


def query(question):
    """asks the chatbot a question, performs RAG, returns the entire answer at once (no streaming)"""

    docs_and_info = retrieval.retrieve_documents(question, n_docs=20)
    docs_string = retrieval.convert_retrieved_documents_to_string(docs_and_info)

    input = {
        "question": question,
        "retrieved_documents": docs_string
    }

    output = QUERY_CHAIN.invoke(
        input=input
    )

    return output


async def query_stream(question):
    """asks the chatbot a question, performs RAG, and streams the answer token by token"""

    docs_and_info = retrieval.retrieve_documents(question)
    docs_string = retrieval.convert_retrieved_documents_to_string(docs_and_info)

    input = {
        "question": question,
        "retrieved_documents": docs_string
    }

    generator = QUERY_CHAIN.astream_events(
        input=input,
        version="v1"
    )

    async for chunk in generator:
        yield f"data: {chunk}\n\n"
