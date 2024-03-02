from flask import Flask, request
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

app = Flask(__name__)

def model(imageUrl: str, jobTitle: str):
    # prepare image + question
    url = imageUrl

    image = Image.open(requests.get(url, stream=True).raw)  # image = Image.open(image_path) if the image is local
    image = image.resize((640, 480))
    image = image.convert('RGB')

    text = f"Given a screenshot of a person's computer, can you determine if the individual appears to be working knowing that the person is a {jobTitle} (respond with yes or no) ?"

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]

@app.route("/", methods=["POST"])
def classify_image():
    data = request.json
    img_url = data.get("img_url")
    job_title = data.get("job_title")
    
    if img_url and job_title:
        result = model(imageUrl=img_url, jobTitle=job_title)
        return {"result": result}
    else:
        return {"error": "Invalid request parameters"}

if __name__ == "__main__":
    app.run(port=8000)  # Change the port number as needed.
