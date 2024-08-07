from sentence_transformer_langchain_wrapper import SentenceTransformerEmbeddings
import chromadb

EMBEDDINGS_GIST = SentenceTransformerEmbeddings(
    model="avsolatorio/GIST-small-Embedding-v0"
)

CHROMADB_CLIENT = chromadb.PersistentClient(
    path="chroma_db"
).get_or_create_collection(
    name="rag",
    metadata={"hnsw:space": "cosine"}
)


def retrieve_documents(question, n_docs=10):
    """takes in a string query, then retrieves related documents inside the chromadb storage"""

    question_embed = EMBEDDINGS_GIST.embed_query(question)

    results = CHROMADB_CLIENT.query(
        query_embeddings=[question_embed],
        n_results=n_docs
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]


    # create a list of 3-tuple (document, score, source) for every retrieved documents
    # the score is 1 - distance, rounded to 4 decimal points
    docs_and_info = []
    for i in range(len(documents)):
        doc = documents[i]
        score = round(1 - distances[i], 4)
        source = metadatas[i]["source"]
        docs_and_info.append(
            (doc, score, source)
        )

    return docs_and_info


def convert_retrieved_documents_to_string(docs_and_info):
    """converts docs_and_info into a single string for LLM input augmentation"""

    output_string = []
    for i, (doc, score, source) in enumerate(docs_and_info):
        output_string.append(f"""
Document {i+1}:
- Source: {source}
- Confidence Score: {score}
- Text Content: {doc}
""")
    
    output_string = "".join(output_string)

    return output_string
