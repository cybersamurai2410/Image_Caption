from transformers import AutoProcessor, BlipForQuestionAnswering
from PIL import Image

# Load model and processor
model = BlipForQuestionAnswering.from_pretrained("./models/Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("./models/Salesforce/blip-vqa-base")

def visual_qa(image, question):
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
