from sentence_transformer_langchain_wrapper import SentenceTransformerEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import chromadb
import os

EMBEDDINGS_GIST = SentenceTransformerEmbeddings(
    model="avsolatorio/GIST-small-Embedding-v0"
)

CHROMADB_CLIENT = chromadb.PersistentClient(
    path="chroma_db"
).get_or_create_collection(
    name="rag",
    metadata={"hnsw:space": "cosine"}
)


def insert_document_to_vectordb(filename, verbose=False):
    """inserts text document into chromadb vector database"""

    # open file text content
    document = None
    with open(f"./documents/{filename}", "r") as file:
        text = file.read()
        document = Document(
            page_content=text,
            metadata={"source": filename}
        )
    
    if verbose:
        print(f"splitting {filename} into chunks...")

    # split text into chunks
    splitter = SemanticChunker(
        embeddings=EMBEDDINGS_GIST,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=-2
    )

    # get text content of document chunks
    chunks = splitter.split_documents([document])
    chunks = [chunk.page_content for chunk in chunks]

    # convert text chunks into vector embeddings
    embeddings = EMBEDDINGS_GIST.embed_documents(chunks)
    
    if verbose:
        print(f"inserting {filename} chunks into chromadb...")

    # upsert (update if exists, insert if not) vector embeddings into chromadb storage
    CHROMADB_CLIENT.upsert(
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": filename}] * len(chunks),
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
    )
    
    if verbose:
        print(f"finished inserting {filename} chunks into chromadb with chunk count {len(chunks)}")
        print("----------------------------------------------------------------------------------")


# example usage
def main():
    documents = os.listdir("./documents/")

    for doc in documents:
        insert_document_to_vectordb(doc, verbose=True)


if __name__ == "__main__":
    main()