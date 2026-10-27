import logging

from sentence_transformers import SentenceTransformer

from database import get_connection


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def search_similar_documents(query, model_name, top_k=3):
    """
    Hybrid Search:
    1. Vector/semantic search using pgvector.
    2. Keyword search using PostgreSQL Full-Text Search.
    3. Combine both rankings using Reciprocal Rank Fusion (RRF).
    """

    logger.info("Starting hybrid search")
    logger.info("Query: %s", query)

    # 1. Generate query embedding

    logger.info("Loading embedding model: %s", model_name)

    model = SentenceTransformer(model_name)

    logger.info("Generating query embedding")

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    logger.info(
        "Query embedding generated with shape: %s",
        query_embedding.shape
    )

    # 2. Search PostgreSQL

    connection = get_connection()

    try:

        with connection.cursor() as cursor:

            # VECTOR SEARCH
            
            vector_k = max(top_k * 3, 10)

            logger.info(
                "Running vector search: top %d",
                vector_k
            )

            cursor.execute(
                """
                SELECT
                    id,
                    content,
                    metadata,
                    1 - (embedding <=> %s) AS similarity
                FROM rag_documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (
                    query_embedding,
                    query_embedding,
                    vector_k,
                ),
            )

            vector_results = cursor.fetchall()

            logger.info(
                "Vector search returned %d results",
                len(vector_results)
            )

            # KEYWORD SEARCH

            logger.info(
                "Running keyword search: top %d",
                vector_k
            )

            cursor.execute(
                """
                SELECT
                    id,
                    content,
                    metadata,
                    ts_rank(
                        search_vector,
                        plainto_tsquery('english', %s)
                    ) AS keyword_score
                FROM rag_documents
                WHERE search_vector @@ plainto_tsquery(
                    'english',
                    %s
                )
                ORDER BY keyword_score DESC
                LIMIT %s;
                """,
                (
                    query,
                    query,
                    vector_k,
                ),
            )

            keyword_results = cursor.fetchall()

            logger.info(
                "Keyword search returned %d results",
                len(keyword_results)
            )

        # 3. Reciprocal Rank Fusion (RRF)

        logger.info("Combining results using RRF")

        rrf_k = 60

        combined_results = {}

        # Vector results
        for rank, result in enumerate(vector_results, start=1):

            document_id = result[0]

            if document_id not in combined_results:

                combined_results[document_id] = {
                    "id": document_id,
                    "content": result[1],
                    "metadata": result[2],
                    "rrf_score": 0.0,
                }

            combined_results[document_id]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

        # Keyword results
        for rank, result in enumerate(keyword_results, start=1):

            document_id = result[0]

            if document_id not in combined_results:

                combined_results[document_id] = {
                    "id": document_id,
                    "content": result[1],
                    "metadata": result[2],
                    "rrf_score": 0.0,
                }

            combined_results[document_id]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

        # 4. Sort by final RRF score

        final_results = sorted(
            combined_results.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        final_results = final_results[:top_k]

        logger.info(
            "Hybrid search returned %d final results",
            len(final_results)
        )

        # 5. Return in same format as previous retrieval.py

        return [
            (
                result["id"],
                result["content"],
                result["metadata"],
                result["rrf_score"],
            )
            for result in final_results
        ]

    finally:

        connection.close()

        logger.info("PostgreSQL connection closed")


if __name__ == "__main__":

    query = "What is self-attention?"

    logger.info("Testing hybrid search")

    results = search_similar_documents(
        query,
        "sentence-transformers/all-MiniLM-L6-v2",
        top_k=3
    )

    logger.info("Displaying hybrid search results")

    for result in results:

        document_id = result[0]
        content = result[1]
        metadata = result[2]
        rrf_score = result[3]

        logger.info(
            "Document ID: %s | Page: %s | RRF Score: %.6f",
            document_id,
            metadata.get("page") if metadata else None,
            rrf_score
        )

        logger.info("Content:\n%s", content)
