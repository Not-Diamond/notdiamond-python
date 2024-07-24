import os

import pytest
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from notdiamond.llms.client import NotDiamond


@pytest.mark.skip(reason="Skipping due to lack of access to Ada")
def test_rag_with_llamaindex():
    # Define the user query
    query = "What is the cancellation policy?"

    # Load the terms of service documents and pass them into a vector store for retrieval
    documents = SimpleDirectoryReader(
        os.getcwd() + "/tests/static"
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever()

    # Retrieve the relevant context from the terms of service
    nodes = retriever.retrieve(query)
    context = ""
    for node in nodes:
        context += f"{node.get_text()}\n"

    # Define your prompt template
    prompt = f"""
      You are helping a customer with the terms of service.
      The following document is the relevant part of the terms of service to the query.
      Document: {context}
      Query: {query}
      """

    # Define the LLMs you'd like to route between
    llm_configs = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-3-opus-20240229",
    ]

    # Create the NDLLM object
    nd_llm = NotDiamond(llm_configs=llm_configs)

    # After fuzzy hashing, the best LLM is determined by the ND API and the LLM called client-side
    result, session_id, provider = nd_llm.invoke(prompt)

    print("ND session ID: ", session_id)  # A unique ID of the invoke
    print("LLM called: ", provider.model)  # The LLM routed to
    print("LLM output: ", result.content)  # The LLM response
