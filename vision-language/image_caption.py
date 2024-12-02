from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model and processor
model = BlipForConditionalGeneration.from_pretrained("./models/Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("./models/Salesforce/blip-image-captioning-base")

def image_captioning(image):
    inputs = processor(image, "a photograph of", return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
