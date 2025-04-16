import torch
from torchvision import models, transforms
from PIL import Image

class_labels = {0: "Normal", 1: "Tuberculosis"}



def load_model(model_path):
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)  # FIX: match training setup
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    with torch.no_grad():
        # Get raw model outputs (logits)
        outputs = model(image_tensor)
        
        print(f"Raw Model Outputs (logits): {outputs}")  # Debugging the raw output
        print(f"Shape of the logits: {outputs.shape}")  # This will tell us the shape of the output

        # Apply softmax to get probabilities
        probs = torch.softmax(outputs, dim=1)  
        print(f"Softmax probabilities: {probs}")  # Debugging the probabilities

        # Get the predicted class index
        predicted_class = torch.argmax(probs, dim=1).item()
        print(f"Predicted Class Index: {predicted_class}")  # Debugging the predicted class

        # Check if the predicted class is in the class_labels
        if predicted_class not in class_labels:
            print(f"Warning: Predicted class {predicted_class} is not in class_labels")
            return "Unknown", probs.numpy().squeeze()

        return class_labels[predicted_class], probs.numpy().squeeze()


# Example usage
# Ensure to use the correct model path
model = load_model("model/01_tuberculosis_model (2).pth")


# Example image path
image_path = 'C:/path/to/your/image.jpg'
image = Image.open(image_path)
image_tensor = preprocess_image(image)

label, probs = predict(model, image_tensor)
print(f"✅ Prediction: {label}")
print(f"📊 Confidence Scores\n{probs}")
