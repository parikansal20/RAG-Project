import os

from dotenv import load_dotenv
from exa_py import Exa

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

exa = Exa(
    api_key=os.getenv("EXA_API_KEY")
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# Langchain Prompt

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the user's question using the web search
results provided below.

Question:
{query}

Web search results:
{context}

Instructions:
- Use the provided web results to answer the question.
- Do not make up information.
- If the results do not contain enough information,
  clearly say so.
- Give a clear and concise answer.
- Mention the relevant source URLs when appropriate.
""")

chain = prompt | llm

#Web search using Exa

def web_search(query):

    response = exa.search(
        query,
        type="auto",
        num_results=5,
        contents={
            "highlights": True
        }
    )

    return response.results

# 7.Generating answer using Gemini

def generate_answer(query, results):

    context = ""

    for i, result in enumerate(results, start=1):

        context += f"""
SOURCE {i}

Title:
{result.title}

URL:
{result.url}

Relevant content:
{result.highlights}

--------------------------------
"""

    # Send query + Exa context to LangChain
    response = chain.invoke({
        "query": query,
        "context": context
    })

    return response.content

if __name__ == "__main__":

    query = input("Ask a question: ")

    # Step 1: Search the web using Exa
    results = web_search(query)

    # Step 2: Give Exa results to Gemini
    answer = generate_answer(query, results)

    # Step 3: Print final answer
    print("\nFINAL ANSWER:\n")
    print(answer)
