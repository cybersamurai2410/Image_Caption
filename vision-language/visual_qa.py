from transformers import AutoProcessor, BlipForQuestionAnswering

vqa_id = "Salesforce/blip-vqa-base"
vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_id)
vqa_processor = AutoProcessor.from_pretrained(vqa_id)

def visual_qa(image, question):
    inputs = vqa_processor(image, question, return_tensors="pt")
    out = vqa_model.generate(**inputs)
    return vqa_processor.decode(out[0], skip_special_tokens=True)
