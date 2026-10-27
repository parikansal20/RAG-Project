import logging
import os

from retrieval import search_similar_documents
from generation import generate_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.6-flash"
TOP_K = 3


def rag_pipeline(query, embedding_model, llm_model, api_key, top_k):
    """
    Run the complete RAG pipeline.

    1. Retrieve relevant documents.
    2. Generate an answer using the retrieved context.
    """

    logger.info("Starting RAG pipeline")
    logger.info("User query: %s", query)


    retrieved_documents = search_similar_documents(
        query,
        embedding_model,
        top_k
    )

    logger.info(
        "Retrieved %d documents",
        len(retrieved_documents)
    )


    answer = generate_answer(
        query,
        retrieved_documents,
        api_key,
        llm_model
    )

    logger.info("RAG pipeline completed successfully")

    return answer


if __name__ == "__main__":

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set")
        raise ValueError("GEMINI_API_KEY environment variable is required")

    query = input("Ask a question: ")

    answer = rag_pipeline(
        query,
        EMBEDDING_MODEL,
        LLM_MODEL,
        api_key,
        TOP_K
    )

    logger.info("Final answer:\n%s", answer)
