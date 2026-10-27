import logging
from pypdf import PdfReader


logger = logging.getLogger(__name__)

PDF_PATH = "data/raw/transformers_explained.md.pdf"


def load_pdf(pdf_path):
    """
    Load a PDF and extract text from every page.
    """

    logger.info("Loading PDF: %s", pdf_path)

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            documents.append({
                "page": page_number + 1,
                "content": text
            })

            logger.info(
                "Extracted text from page %d",
                page_number + 1
            )
        else:
            logger.warning(
                "No text found on page %d",
                page_number + 1
            )

    logger.info(
        "PDF loading completed. Pages with text: %d",
        len(documents)
    )

    return documents


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    documents = load_pdf(PDF_PATH)

    logger.info("Number of pages with text: %d", len(documents))

    for document in documents[:2]:

        logger.info(
            "PAGE %d | Content preview: %s",
            document["page"],
            document["content"][:1000]
        )
    