import logging

from google import genai


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def generate_answer(query, retrieved_documents, api_key, model_name):
    """
    Generate an answer using the user's query
    and the retrieved document chunks.
    """

    logger.info("Starting answer generation")

    # Create context from retrieved documents
    context_parts = []

    for document in retrieved_documents:

        document_id = document[0]
        content = document[1] 
        metadata = document[2]

        page = metadata.get("page", "Unknown")

        context_parts.append(
            f"""
Document ID: {document_id}
Page: {page}

Content:
{content}
"""
        )

    context = "\n".join(context_parts)

    logger.info(
        "Context created from %d retrieved documents",
        len(retrieved_documents)
    )


    prompt = f"""
You are a helpful assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

User Question:
{query}

Answer:
"""

    logger.info("Sending prompt to LLM")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    logger.info("Answer generated successfully")

    return response.text

if __name__ == "__main__":

    logger.info("Generator module loaded successfully")
