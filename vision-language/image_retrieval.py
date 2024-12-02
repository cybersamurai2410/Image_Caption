from transformers import AutoProcessor, BlipForImageTextRetrieval
from PIL import Image
import requests
import torch

# Load model and processor
model = BlipForImageTextRetrieval.from_pretrained("./models/Salesforce/blip-itm-base-coco")
processor = AutoProcessor.from_pretrained("./models/Salesforce/blip-itm-base-coco")

def image_retrieval(image_url, text):
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    inputs = processor(images=raw_image, text=text, return_tensors="pt")
    itm_scores = model(**inputs)[0]
    itm_score = torch.nn.functional.softmax(itm_scores, dim=1)
    probability = itm_score[0][1].item()
    return f"The image and text are matched with a probability of {probability:.4f}"
