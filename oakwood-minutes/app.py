"""A simple web UI for residents."""

import gradio as gr

from ask import check_setup, stream_answer


def respond(message, history):
    """Stream the answer into the chat as it's written."""
    text = ""
    for piece in stream_answer(message):
        text += piece
        yield text


demo = gr.ChatInterface(
    fn=respond,
    title="Ask Oakwood City Council (Oakwood minutes demo)",
    description="Answers come only from published meeting records, with citations. "
    "Always check the original document before relying on an answer.",
    examples=[
        "What did the council decide about the pickleball court?",
        "What was approved for the street lighting program?",
        "How much did the city spend on parks last year?",
    ],
)

if __name__ == "__main__":
    check_setup()
    demo.launch()
