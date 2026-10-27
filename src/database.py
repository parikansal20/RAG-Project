import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

load_dotenv()


def get_connection():
    """
    Create and return a connection to PostgreSQL.
    """

    connection = psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD") or None,
    )

    # Tell psycopg how to work with pgvector
    register_vector(connection)

    return connection


def test_connection():
    """
    Test the PostgreSQL connection.
    """

    try:
        connection = get_connection()

        print("Successfully connected to PostgreSQL!")

        connection.close()

    except Exception as e:
        print("PostgreSQL connection failed.")
        print(f"Error: {e}")


def insert_documents(chunks, embeddings):
    """
    Insert chunks, metadata, and embeddings into PostgreSQL.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must be the same."
        )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            for chunk, embedding in zip(chunks, embeddings):

                cursor.execute(
                    """
                    INSERT INTO rag_documents
                    (content, metadata, embedding)
                    VALUES (%s, %s, %s);
                    """,
                    (
                        chunk["content"],
                        Jsonb(chunk["metadata"]),
                        embedding,
                    ),
                )

        connection.commit()

        logger.info(f"Inserted %d documents into PostgreSQL.", len(chunks))

    except Exception as e:

        connection.rollback()

        print("Failed to insert documents.")
        print(f"Error: {e}")

        raise

    finally:
        connection.close()


def get_documents():
    """
    Retrieve documents from PostgreSQL.
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT id, content, metadata
                FROM rag_documents
                ORDER BY id;
                """
            )

            documents = cursor.fetchall()

        return documents

    finally:
        connection.close()


if __name__ == "__main__":

    test_connection()

    documents = get_documents()

    print(f"\nNumber of documents: {len(documents)}")

    for document in documents:

        print("\n" + "=" * 60)
        print("ID:", document[0])
        print("Content:", document[1][:200])
        print("Metadata:", document[2]) 