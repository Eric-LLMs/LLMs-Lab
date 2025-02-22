import sys

import gradio as gr
import pandas as pd

from AssistantGPT import AssistantGPT
from config import DEFAULT_MODEL, MODEL_TO_MAX_TOKENS, MODELS, DEFAULT_MAX_TOKENS
from file_processor_helper import FileProcessorHelper
from loguru import logger
from utils import build_chat_document_prompt, upload_files

logger.remove()  # Remove the default handler added after importing logger to prevent duplicate outputs
logger.add(sys.stderr, level="DEBUG")  # Adjust log output level: INFO|DEBUG|TRACE


def fn_update_max_tokens(model, origin_set_tokens):
    """
    Function to update the maximum number of tokens.

    :param model: The model for which the maximum number of tokens needs to be updated.
    :param origin_set_tokens: The number of tokens set in the original slider component.
    :return: A slider component with the new maximum number of tokens.
    """
    # Get the new maximum number of tokens for the model, defaulting to the provided max tokens if not set
    new_max_tokens = MODEL_TO_MAX_TOKENS.get(model, origin_set_tokens)

    # If the original setting exceeds the new maximum, adjust it to the default value
    new_set_tokens = origin_set_tokens if origin_set_tokens <= new_max_tokens else DEFAULT_MAX_TOKENS

    # Create a new max tokens slider component
    new_max_tokens_component = gr.Slider(
        minimum=0,
        maximum=new_max_tokens,
        value=new_set_tokens,
        step=1.0,
        label="max_tokens",
        interactive=True,
    )

    return new_max_tokens_component


def fn_prehandle_user_input(user_input, chat_history):
    logger.info(f"Component input | user_input: {user_input} chat_history: {chat_history}")

    # Initialize
    chat_history = [] if not chat_history else chat_history

    # Check input
    if not user_input:
        gr.Warning("Please enter your question")
        logger.warning("Please enter your question")
        return chat_history

    # Display user message in the chatbox
    chat_history.append([user_input, None])

    return chat_history


def fn_chat(chat_mode, uploaded_file_paths_df, user_input, chat_history, model, max_tokens, temperature, stream, top_n):
    if not user_input:
        return chat_history

    uploaded_file_paths = uploaded_file_paths_df['Uploaded Files'].values.tolist()

    logger.info(f"\n"
                f"Chat Mode: {chat_mode} \n"
                f"File Paths: {uploaded_file_paths} {type(uploaded_file_paths)} \n"
                f"User Input: {user_input} \n"
                f"Chat History: {chat_history} \n"
                f"Model Used: {model} {type(model)}\n"
                f"Max Tokens: {max_tokens} {type(max_tokens)}\n"
                f"Temperature: {temperature} {type(temperature)}\n"
                f"Stream Output: {stream} {type(stream)}\n"
                f"Top N: {top_n} {type(top_n)}")

    messages = []
    if chat_mode == "Normal Q&A":
        messages = user_input if len(chat_history) <= 1 else [
            {"role": "user", "content": chat[0]} if chat[0] else None,
            {"role": "assistant", "content": chat[1]} if chat[1] else None
        ]
    else:
        if not isinstance(uploaded_file_paths, list) or not uploaded_file_paths or '' in uploaded_file_paths:
            gr.Warning("No files uploaded")
            return chat_history

        user_prompt = build_chat_document_prompt(uploaded_file_paths, user_input, chat_history, top_n)
        messages.append({"role": "user", "content": user_prompt}) if user_prompt else logger.error(
            "Failed to generate user_prompt")

    if not messages:
        logger.error("messages is empty")
        gr.Warning("Server error")
        return chat_history

    gpt = AssistantGPT()
    bot_response = gpt.get_completion(messages, model, max_tokens, temperature, stream)

    if stream:
        chat_history[-1][1] = ""
        for character in bot_response:
            char_content = character.choices[0].delta.content
            if char_content is not None:
                chat_history[-1][1] += char_content
                yield chat_history
            else:
                logger.success(f"Streaming Output | bot_response: {chat_history[-1][1]}")
    else:
        chat_history[-1][1] = bot_response
        logger.success(f"Non-streaming Output | bot_response: {chat_history[-1][1]}")
        yield chat_history


def fn_upload_files(unuploaded_file_paths):
    logger.trace(f"Component input | unuploaded_file_paths: {unuploaded_file_paths}")
    uploaded_file_paths = []

    for file_path in unuploaded_file_paths:
        result = upload_files(str(file_path))
        if result.get('code') == 200:
            gr.Info("File uploaded successfully!")
            uploaded_file_paths.append(result.get('data').get('uploaded_file_path'))
        else:
            raise gr.Error("File upload failed!")

    return pd.DataFrame({'Uploaded Files': uploaded_file_paths})


with gr.Blocks() as demo:
    gr.Markdown("# <center>AssistantGPT</center>")
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Chatbot")
            user_input_textbox = gr.Textbox(label="User Input", value="What is this article about?")
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear")
        with gr.Column(scale=1):
            with gr.Tab(label="Q&A"):
                chat_mode_radio = gr.Radio(["Normal Q&A", "Document Q&A"], label="Q&A Mode", value="Document Q&A",
                                           interactive=True)
                file_paths_files = gr.Files(label="Upload Files", file_count="multiple", file_types=[".pdf", ".txt"],
                                            type="filepath")
                file_paths_dataframe = gr.Dataframe(value=pd.DataFrame({'Uploaded Files': []}))
                top_n_number = gr.Number(label="top_n", value=20)
            with gr.Tab(label="Model Parameters"):
                model_dropdown = gr.Dropdown(label="model", choices=MODELS, value=DEFAULT_MODEL, multiselect=False,
                                             interactive=True)
                max_tokens_slider = gr.Slider(minimum=0, maximum=4096, value=1000, step=1.0, label="max_tokens",
                                              interactive=True)
                temperature_slider = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.01, label="temperature",
                                               interactive=True)
                stream_radio = gr.Radio([True, False], label="stream", value=True, interactive=True)

demo.queue().launch()
