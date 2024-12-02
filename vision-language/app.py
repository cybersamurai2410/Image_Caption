import gradio as gr
from transformers import AutoProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval, BlipForQuestionAnswering
from PIL import Image
import torch
import requests

# Load models and processors
caption_model = BlipForConditionalGeneration.from_pretrained("./models/Salesforce/blip-image-captioning-base")
caption_processor = AutoProcessor.from_pretrained("./models/Salesforce/blip-image-captioning-base")

retrieval_model = BlipForImageTextRetrieval.from_pretrained("./models/Salesforce/blip-itm-base-coco")
retrieval_processor = AutoProcessor.from_pretrained("./models/Salesforce/blip-itm-base-coco")

vqa_model = BlipForQuestionAnswering.from_pretrained("./models/Salesforce/blip-vqa-base")
vqa_processor = AutoProcessor.from_pretrained("./models/Salesforce/blip-vqa-base")

# Image Captioning
def image_captioning(image):
    inputs = caption_processor(image, "a photograph of", return_tensors="pt")
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

# Image Retrieval
def image_retrieval(image_url, text):
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    inputs = retrieval_processor(images=raw_image, text=text, return_tensors="pt")
    itm_scores = retrieval_model(**inputs)[0]
    itm_score = torch.nn.functional.softmax(itm_scores, dim=1)
    probability = itm_score[0][1].item()
    return f"The image and text are matched with a probability of {probability:.4f}"

# Visual Question Answering
def visual_qa(image, question):
    inputs = vqa_processor(image, question, return_tensors="pt")
    out = vqa_model.generate(**inputs)
    return vqa_processor.decode(out[0], skip_special_tokens=True)

# Gradio interfaces
caption_interface = gr.Interface(
    fn=image_captioning,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning",
    description="Generate a caption for the uploaded image."
)

retrieval_interface = gr.Interface(
    fn=image_retrieval,
    inputs=[
        gr.Textbox(label="Image URL"),
        gr.Textbox(label="Description Text")
    ],
    outputs=gr.Textbox(label="Matching Probability"),
    title="Image Retrieval",
    description="Check if the image and text match semantically."
)

vqa_interface = gr.Interface(
    fn=visual_qa,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Visual Question Answering",
    description="Answer questions about the uploaded image."
)

# Combine into a tabbed interface
app = gr.TabbedInterface(
    interface_list=[caption_interface, retrieval_interface, vqa_interface],
    tab_names=["Image Captioning", "Image Retrieval", "Visual Q&A"]
)

# Launch the app
app.launch()
