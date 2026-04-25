import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

# Load pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

target_layer = model.layer4[-1]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===== GRAD-CAM =====
def generate_heatmap(img, model, target_layer):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hook
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    input_tensor = transform(img).unsqueeze(0)
    input_tensor.requires_grad = True

    output = model(input_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0].detach().numpy()[0]
    act = activations[0].detach().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    if cam.max() != 0:
        cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # remove hook biar tidak numpuk
    handle_f.remove()
    handle_b.remove()

    return overlay

# ===== MAIN =====
def predict_image(filepath):
    try:
        img = cv2.imread(filepath)

        if img is None:
            return "ERROR", 0, {}, None, "Gambar tidak terbaca"

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            prob = probs.max().item()

        real_score = prob
        fake_score = 1 - prob

        if prob > 0.7:
            result = "REAL"
            confidence = round(prob * 100, 2)
            reason = "Citra terlihat natural"
        else:
            result = "FAKE"
            confidence = round(prob * 100, 2)
            reason = "Kemungkinan manipulasi visual"

        breakdown = {
            "real": round(real_score * 100, 2),
            "fake": round(fake_score * 100, 2)
        }

        # ===== HEATMAP =====
        heatmap_img = generate_heatmap(img, model, target_layer)

        os.makedirs("static", exist_ok=True)
        heatmap_path = "heatmap.jpg"
        cv2.imwrite(os.path.join("static", heatmap_path), heatmap_img)

        return result, confidence, breakdown, heatmap_path, reason

    except Exception as e:
        return "ERROR", 0, {}, None, str(e)