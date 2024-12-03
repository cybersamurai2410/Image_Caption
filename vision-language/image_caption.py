from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image

caption_id = "Salesforce/blip-image-captioning-base"
caption_model = BlipForConditionalGeneration.from_pretrained(caption_id)
caption_processor = AutoProcessor.from_pretrained(caption_id)

def image_captioning(image):
    inputs = processor(image, "a photograph of", return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
