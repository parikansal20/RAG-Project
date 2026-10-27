import logging

from sentence_transformers import SentenceTransformer

from chunking import create_chunks
from document_loader import load_pdf
from database import insert_documents


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


PDF_PATH = "data/raw/transformers_explained.md.pdf"


def generate_embeddings(chunks, model_name):
    """
    Convert text chunks into numerical embeddings.
    """

    logger.info("Loading embedding model: %s", model_name)

    model = SentenceTransformer(model_name)

    logger.info("Number of chunks to embed: %d", len(chunks))

    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    logger.info("Embeddings generated successfully")
    logger.info("Embedding shape: %s", embeddings.shape)

    return embeddings


if __name__ == "__main__":

    logger.info("Starting RAG ingestion pipeline")

    logger.info("Loading PDF: %s", PDF_PATH)
    documents = load_pdf(PDF_PATH)

    logger.info("Creating chunks")
    chunks = create_chunks(documents)

    logger.info("Chunks created: %d", len(chunks))

    embeddings = generate_embeddings(
        chunks,
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    logger.info("Inserting documents into PostgreSQL")
    insert_documents(chunks, embeddings)

    logger.info("RAG ingestion completed successfully")

