"""
Simple web UI for the Technical Documentation Assistant
"""

import os
import sys
import gradio as gr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(use_llm=False)  # Set to True if you want to use LLM

def process_query(query):
    """Process a query and return the response"""
    return pipeline.process_query(query)

# Create Gradio interface
with gr.Blocks(title="Technical Documentation Assistant") as demo:
    gr.Markdown("# Technical Documentation Assistant")
    gr.Markdown("Ask questions about PyTorch, TensorFlow, and Hugging Face Transformers")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about a technical topic...",
                lines=2
            )
            submit_btn = gr.Button("Submit")
        
    with gr.Row():
        response_output = gr.Textbox(
            label="Answer",
            lines=10,
            interactive=False
        )
    
    submit_btn.click(
        fn=process_query,
        inputs=query_input,
        outputs=response_output
    )

if __name__ == "__main__":
    demo.launch(share=False)
