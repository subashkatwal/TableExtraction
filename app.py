import uuid
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection, pipeline
from pytesseract import image_to_string
import os
from werkzeug.utils import secure_filename
from PIL import ImageFilter




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")


os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "files" not in request.files:
            return "No file part"
        file = request.files["files"]
        if not file.filename:
            return "No file selected"
        
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(file_path)

        if not os.path.exists(file_path):
            return f"File not found at {file_path}"

        # Process the image for table detection
        image = Image.open(file_path)

        image = image.filter(ImageFilter.SHARPEN)

        # Convert the image to RGB if it's not already in that mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = processor(images=image, return_tensors="pt", size={"longest_edge": 1000, "shortest_edge": 800})

        outputs = model(**inputs)
        results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=[image.size])

        # Initialize extracted_text variable
        extracted_text = ""

        for result in results:
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                label_value = label.item()  # Access label value
                xmin, ymin, xmax, ymax = map(int, box.tolist())
                cropped_table = image.crop((xmin, ymin, xmax, ymax))
                extracted_text = image_to_string(cropped_table, lang="eng+nep", config="--oem 1 --psm 6")


        # Store extracted text for QA
        with open("extracted_text.txt","w", encoding="utf-8") as f:
            f.write(extracted_text)

        return render_template("index.html", extracted_text=extracted_text)

    return render_template("index.html", extracted_text=None)
@app.route("/ask", methods=["POST"])
def ask():
    # Retrieve extracted text
    with open("extracted_text.txt", "r") as f:
        context = f.read()

    # Check if 'question' is present in the form
    if "question" not in request.form:
        return "Error: No question provided in the form submission", 400

    # Get the user question from the form
    question = request.form["question"]
    answer = qa_pipeline(question=question, context=context)

    return render_template("index.html", extracted_text=context, question=question, answer=answer["answer"])
if __name__ == "__main__":
    app.run(debug=True)

