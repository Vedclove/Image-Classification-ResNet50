from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

app = Flask(__name__)

# ✅ Load Model at Startup
MODEL_PATH = "resnet50_mps.pth"
device = torch.device("cpu")  # Use CPU

# Create a ResNet50 model instance
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(2048, 2)  # Modify final layer for 2 classes (cat, dog)

# Load state dictionary and move to CPU
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ✅ Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/")
def home():
    return jsonify({"message": "Flask server is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    class_names = ['Cat', 'Dog']
    result = class_names[predicted.item()]
    
    return jsonify({"predicted_class": result})

# ✅ Run the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)