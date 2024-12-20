import gradio as gr
from image_caption import image_captioning
from image_retrieval import image_retrieval
from visual_qa import visual_qa

caption_interface = gr.Interface(
    fn=image_captioning,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning",
    description="Generate a caption for the uploaded image.",
    allow_flagging="never"  
)

retrieval_interface = gr.Interface(
    fn=image_retrieval,
    inputs=[
        gr.Textbox(label="Image URL"),
        gr.Textbox(label="Description Text")
    ],
    outputs=[
        gr.Image(label="Retrieved Image"),  
        gr.Textbox(label="Matching Probability")
    ],
    title="Image Retrieval",
    description="Check if the image and text match semantically.",
    allow_flagging="never"  
)

vqa_interface = gr.Interface(
    fn=visual_qa,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Visual Question Answering",
    description="Answer questions about the uploaded image.",
    allow_flagging="never"  
)

# Combine vision-langauge tasks into a tabbed interface
app = gr.TabbedInterface(
    interface_list=[caption_interface, retrieval_interface, vqa_interface],
    tab_names=["Image Captioning", "Image Retrieval", "Visual Q&A"]
)

app.launch()
