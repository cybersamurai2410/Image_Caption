from transformers import AutoProcessor, BlipForConditionalGeneration

caption_id = "Salesforce/blip-image-captioning-base"
caption_model = BlipForConditionalGeneration.from_pretrained(caption_id)
caption_processor = AutoProcessor.from_pretrained(caption_id)

def image_captioning(image):
    inputs = caption_processor(image, "a photograph of", return_tensors="pt")
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)
