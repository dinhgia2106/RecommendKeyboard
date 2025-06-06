#!/usr/bin/env python3
"""
Simple test script để check Gradio events
"""

import gradio as gr
import time


def simple_segment(text, show_multiple):
    """Simple segmentation function for testing"""
    if not text:
        return "", 0.0, "Nhập text để test", ""
    
    # Simulate processing
    time.sleep(0.01)
    
    # Simple fake segmentation
    segmented = " ".join(text)  # Add space between each character
    
    status = f"Processed '{text}' in 0.01s | Length: {len(text)} | Auto-update: {'ON' if True else 'OFF'}"
    
    suggestions = ""
    if show_multiple:
        suggestions = f"Suggestions:\n1. {segmented}\n2. {text[::-1]}\n3. {text.upper()}"
    
    return segmented, 0.01, status, suggestions


# Create simple Gradio interface
with gr.Blocks(title="Event Test") as demo:
    gr.Markdown("# 🧪 Event Test Interface")
    gr.Markdown("Test real-time events in Gradio")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Type something...",
                lines=2
            )
            show_multiple = gr.Checkbox(
                label="Show multiple suggestions",
                value=True
            )
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Output",
                lines=2,
                interactive=False
            )
            processing_time = gr.Number(
                label="Processing Time",
                interactive=False
            )
    
    status_text = gr.Textbox(
        label="Status",
        interactive=False
    )
    
    suggestions_text = gr.Textbox(
        label="Multiple Suggestions",
        lines=4,
        interactive=False
    )
    
    # Event handlers
    input_text.change(
        fn=simple_segment,
        inputs=[input_text, show_multiple],
        outputs=[output_text, processing_time, status_text, suggestions_text]
    )
    
    input_text.input(
        fn=simple_segment,
        inputs=[input_text, show_multiple],
        outputs=[output_text, processing_time, status_text, suggestions_text]
    )
    
    show_multiple.change(
        fn=simple_segment,
        inputs=[input_text, show_multiple],
        outputs=[output_text, processing_time, status_text, suggestions_text]
    )


if __name__ == "__main__":
    print("🧪 Starting Event Test Demo...")
    print("💡 This will test if Gradio events work properly")
    demo.launch(server_port=7863, share=False) 