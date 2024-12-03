from transformers import AutoProcessor, BlipForQuestionAnswering
from PIL import Image

vqa_id = "Salesforce/blip-vqa-base"
vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_id)
vqa_processor = AutoProcessor.from_pretrained(vqa_id)

def visual_qa(image, question):
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
