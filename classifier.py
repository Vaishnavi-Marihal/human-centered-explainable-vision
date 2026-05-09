import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


# Load pretrained ResNet50 once at module level
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# ImageNet class labels
from torchvision.models import ResNet50_Weights
LABELS = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(image_path: str) -> tuple:
    """
    Load image from path.
    Returns PIL image and preprocessed tensor.
    """
    pil_image = Image.open(image_path).convert("RGB")
    tensor = preprocess(pil_image).unsqueeze(0)  # add batch dim
    return pil_image, tensor


def classify(tensor: torch.Tensor) -> tuple:
    """
    Run inference on preprocessed tensor.
    Returns:
        top5_labels  : list of 5 class name strings
        top5_probs   : list of 5 probability floats
        top5_indices : list of 5 class index integers
        full_probs   : full softmax probability tensor (1000,)
    """
    with torch.no_grad():
        logits = model(tensor)
        full_probs = F.softmax(logits, dim=1).squeeze()

    top5_probs, top5_indices = torch.topk(full_probs, 5)
    top5_labels = [LABELS[i] for i in top5_indices.tolist()]
    top5_probs_list = top5_probs.tolist()

    return top5_labels, top5_probs_list, top5_indices.tolist(), full_probs


def get_model():
    """Return model reference for Grad-CAM use."""
    return model