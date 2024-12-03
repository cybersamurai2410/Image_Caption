from transformers import AutoProcessor, BlipForImageTextRetrieval
from PIL import Image
import requests
import torch

retrieval_id = "Salesforce/blip-itm-base-coco"
retrieval_model = BlipForImageTextRetrieval.from_pretrained(retrieval_id)
retrieval_processor = AutoProcessor.from_pretrained(retrieval_id)

def image_retrieval(image_url, text):
    try:
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
        inputs = retrieval_processor(images=raw_image, text=text, return_tensors="pt")
        itm_scores = retrieval_model(**inputs)[0]
        itm_score = torch.nn.functional.softmax(itm_scores, dim=1)
        probability = itm_score[0][1].item()
        
        return raw_image, f"The image and text are matched with a probability of {probability:.4f}"
    except Exception as e:
        return None, f"Error: {str(e)}"
        
