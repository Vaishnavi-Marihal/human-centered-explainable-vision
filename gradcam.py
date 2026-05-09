import torch
import torch.nn.functional as F
import numpy as np
import cv2
from classifier import get_model


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Implemented from first principles following:
    Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from 
    Deep Networks via Gradient-based Localization"
    
    For a target class c:
        alpha_k^c = (1/Z) * sum_ij (d y^c / d A^k_ij)
        L^c = ReLU( sum_k alpha_k^c * A^k )
    
    Where:
        A^k    : feature map k from target conv layer
        y^c    : score for class c (pre-softmax)
        alpha  : importance weight for each feature map
        ReLU   : retains only positive influences
    """

    def __init__(self):
        self.model = get_model()
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward and backward hooks on ResNet50's 
        final convolutional layer (layer4).
        
        Forward hook  : captures feature map activations
        Backward hook : captures gradients flowing back
        """
        target_layer = self.model.layer4[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def compute(self, tensor: torch.Tensor, class_index: int) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a specific class.
        
        Args:
            tensor      : preprocessed image tensor (1, 3, 224, 224)
            class_index : ImageNet class index to explain
            
        Returns:
            heatmap: numpy array (224, 224) with values in [0, 1]
        """
        self.model.zero_grad()

        # Forward pass — need gradients so no torch.no_grad()
        output = self.model(tensor)

        # Zero all gradients, then backprop only for target class
        self.model.zero_grad()
        
        # Create one-hot score for target class
        score = output[0, class_index]
        score.backward()

        # Global average pooling of gradients over spatial dims
        # Shape: (num_feature_maps,)
        weights = self.gradients.mean(dim=[2, 3]).squeeze()

        # Weighted combination of forward activations
        # Shape: (spatial_h, spatial_w)
        activations = self.activations.squeeze()
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — keep only positive contributions
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input image size (224x224)
        cam_np = cam.numpy()
        heatmap = cv2.resize(cam_np, (224, 224))

        return heatmap

    def overlay_heatmap(
        self, 
        heatmap: np.ndarray, 
        pil_image, 
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            heatmap   : numpy array (224, 224) in [0, 1]
            pil_image : original PIL image
            alpha     : heatmap transparency (0=invisible, 1=opaque)
            
        Returns:
            overlaid: numpy BGR array for display
        """
        # Convert PIL to numpy RGB then BGR for OpenCV
        img_np = np.array(pil_image.resize((224, 224)))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Apply colormap to heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend
        overlaid = cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)

        return overlaid

    def get_peak_region(self, heatmap: np.ndarray) -> str:
        """
        Find spatial location of peak activation.
        Divides image into 3x3 grid and names the region.
        
        Returns human-readable region string e.g. 'upper-left'
        """
        peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        row, col = peak_idx

        row_label = (
            "upper" if row < 75 
            else "lower" if row > 149 
            else "middle"
        )
        col_label = (
            "left" if col < 75 
            else "right" if col > 149 
            else "centre"
        )

        if row_label == "middle" and col_label == "centre":
            return "central region"

        return f"{row_label}-{col_label}"