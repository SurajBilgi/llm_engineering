import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


# Initialize models and tokenizers
def load_model(model_name):
    if model_name == "GPT-2":
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    elif model_name == "Llama-2":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    else:
        model = AutoModelForCausalLM.from_pretrained("gpt2")  # Default model
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer


# Text generation function
def generate_text(prompt, model_name, max_length=100):
    model, tokenizer = load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Custom LLM Models") as demo:
        gr.Markdown("# Custom LLM Models Interface")

        with gr.Row():
            # Side navigation
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=["GPT-2", "Llama-2"], label="Select Model", value="GPT-2"
                )
                max_length = gr.Slider(
                    minimum=50, maximum=500, value=100, step=10, label="Maximum Length"
                )

            # Main content area
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5,
                )
                output_text = gr.Textbox(label="Generated Text", lines=10)
                generate_btn = gr.Button("Generate")

        # Set up the event handler
        generate_btn.click(
            fn=generate_text,
            inputs=[input_text, model_selector, max_length],
            outputs=output_text,
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
