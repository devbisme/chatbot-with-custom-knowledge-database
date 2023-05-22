# MIT License
# Copyright (c) Dave Vandenbout

# This code is a modification of that found in https://beebom.com/how-train-ai-chatbot-custom-knowledge-base-chatgpt-api

from llama_index import (
    load_index_from_storage,
    SimpleDirectoryReader,
    LLMPredictor,
    GPTVectorStoreIndex,
    PromptHelper,
    ServiceContext,
    StorageContext,
)
from langchain.chat_models import ChatOpenAI
import gradio as gr

# os.environ["OPENAI_API_KEY"] = 'Your API Key'

default_docs_dir = "docs"
default_persist_dir = "indexes"


def construct_index(docs_dir=default_docs_dir, persist_dir=default_persist_dir):
    """Construct and store an embedded vector database from the documents in a directory.

    Args:
        docs_dir (str, optional): Path to documents directory. Defaults to default_docs_dir.
        persist_dir (str, optional): Path to directory for storing vector database. Defaults to default_persist_dir.

    Returns:
        GPTVectorStoreIndex: Vector database.
    """

    # Make a PromptHelper that partitions prompts & context into chunks for the LLM.
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    # Make a predictor for how many tokens a given prompt will require.
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs
        )
    )

    # Make a ServiceContext for managing the processing of documents into an embedded vector database.
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    # Create a StorageContext for storing/recalling the vector database.
    storage_context = StorageContext.from_defaults()

    # Create the vector database from the documents.
    documents = SimpleDirectoryReader(docs_dir).load_data()
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context, storage_context=storage_context
    )

    # Store the vector database so it can be re-used without recalculating it.
    storage_context.persist(persist_dir)

    # Return the vector database.
    return index


def chatbot(input_text):
    """Get response to prompt from LLM.

    Args:
        input_text (str): Prompt to LLM.

    Returns:
        str: Response from LLM.
    """
    response = query_engine.query(input_text)
    return response.response


iface = gr.Interface(
    fn=chatbot,
    inputs=gr.components.Textbox(lines=7, label="Enter your text"),
    outputs="text",
    title="Custom-trained AI Chatbot",
)

try:
    # Load embedded vector database from directory.
    storage_context = StorageContext.from_defaults(persist_dir=default_persist_dir)
    index = load_index_from_storage(storage_context=storage_context)
except FileNotFoundError:
    # Vector database not found, so recalculate it.
    index = construct_index()

# Create query interface for vector database.
query_engine = index.as_query_engine()

# Launch web-based chat interface for querying vector database.
iface.launch(share=True)
