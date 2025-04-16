import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# ✅ Confirm correct label order (based on training: 0=TB, 1=Normal)
class_labels = ['Tuberculosis', 'Normal']

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(model_path):
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Predict
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()
        pred_class = class_labels[pred_index]
        return pred_class, probs.detach().cpu().numpy()

# Grad-CAM
def generate_gradcam(model, image_tensor, target_class=None):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)

    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()

    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    forward_handle.remove()
    backward_handle.remove()

    grads_val = gradients[0]
    activations_val = activations[0]
    weights = grads_val.mean(dim=[2, 3], keepdim=True)
    gradcam = F.relu((weights * activations_val).sum(dim=1, keepdim=True))
    gradcam = F.interpolate(gradcam, size=(224, 224), mode='bilinear', align_corners=False)

    gradcam = gradcam.detach().squeeze().cpu().numpy()
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    return heatmap

# Overlay Grad-CAM
def overlay_gradcam(heatmap, image):
    image = image.resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0

    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)

    overlay = heatmap + image_np
    overlay = overlay / overlay.max()
    return np.uint8(255 * overlay)

# Streamlit UI
st.title("🫁 Tuberculosis Detection with Grad-CAM")

model = load_model("model.pth")

uploaded_file = st.file_uploader("📂 Upload a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    image_tensor = preprocess_image(image).to(device)

    label, probs = predict(model, image_tensor)
    st.write(f"🔍 Prediction: **{label}**")
    st.write(f"📊 Confidence Scores: {probs}")

    if label == "Tuberculosis":
        heatmap = generate_gradcam(model, image_tensor)
        overlay = overlay_gradcam(heatmap, image)
        st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
    else:
        st.write("✅ No signs of tuberculosis. Grad-CAM not shown.")

else:
    st.info("Please upload a chest X-ray image to begin diagnosis.")
