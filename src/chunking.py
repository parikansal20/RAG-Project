import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_pdf


logger = logging.getLogger(__name__)

PDF_PATH = "data/raw/transformers_explained.md.pdf"


def create_chunks(documents):
    """
    Split extracted documents into smaller chunks.
    """

    logger.info("Starting document chunking")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = []

    for document in documents:

        page_number = document["page"]

        split_texts = text_splitter.split_text(
            document["content"]
        )

        logger.info(
            "Page %d split into %d chunks",
            page_number,
            len(split_texts)
        )

        for text in split_texts:

            chunks.append({
                "content": text,
                "metadata": {
                    "page": page_number
                }
            })

    logger.info(
        "Chunking completed. Total chunks: %d",
        len(chunks)
    )

    return chunks


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting RAG chunking pipeline")

    documents = load_pdf(PDF_PATH)

    logger.info(
        "Loaded %d documents/pages",
        len(documents)
    )

    chunks = create_chunks(documents)

    logger.info(
        "Number of chunks: %d",
        len(chunks)
    )

    for i, chunk in enumerate(chunks[:5]):

        logger.info(
            "CHUNK %d | Page: %d | Content: %s",
            i + 1,
            chunk["metadata"]["page"],
            chunk["content"]
        )
