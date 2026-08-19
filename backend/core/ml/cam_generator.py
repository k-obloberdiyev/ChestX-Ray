from typing import Tuple, Optional
import logging
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

from backend.config.translations import get_pathology_en, get_pathology_uz
from backend.core.ml.model_manager import get_model, get_device
from backend.core.ml.preprocessor import preprocess_image
from backend.utils import encode_array_to_png

logger = logging.getLogger(__name__)


def generate_gradcam(
    image_bytes: bytes,
    disease: Optional[str] = None,
    model: Optional[xrv.models.DenseNet] = None,
    device: Optional[torch.device] = None
) -> Tuple[bytes, str]:
    """
    Generate a Grad-CAM heatmap overlay for a consumer-selected, auto-selected, or Normal pathology.

    Supports pathology names in English, Uzbek, Russian, or auto-selection.
    Uses direct functional backpropagation (torch.autograd.grad) for clean, leak-free execution.

    Returns:
        tuple[bytes, str]: (PNG encoded image bytes of Grad-CAM overlay, selected disease name)
    """
    if model is None:
        model = get_model()
    if device is None:
        device = get_device()

    # Preprocess image and get original grayscale image
    tensor, orig_gray = preprocess_image(image_bytes, device=device)
    tensor.requires_grad = True

    try:
        # 1. Forward pass extracting convolutional feature maps and unnormalized logits
        features = model.features(tensor)
        out_relu = F.relu(features, inplace=False)
        pooled = F.adaptive_avg_pool2d(out_relu, (1, 1)).view(features.size(0), -1)
        logits = model.classifier(pooled)

        # 2. Resolve disease and target index
        if not disease or not disease.strip():
            # Auto-select pathology with highest prediction score
            target_index = int(torch.argmax(logits[0]).item())
            disease_en = model.pathologies[target_index]
            display_disease = disease_en
            logger.info(f"Auto-selected highest-scoring pathology '{disease_en}' for Grad-CAM.")
        else:
            disease_trimmed = disease.strip()
            disease_en = get_pathology_en(disease_trimmed)
            if disease_en in ["Norma", "Normal"] or disease_trimmed.lower() in ["norma", "normal", "norma (me'yorda)", "норма"]:
                target_index = int(torch.argmax(logits[0]).item())
                disease_en = model.pathologies[target_index]
                display_disease = "Norma"
                logger.info(f"Generated baseline normal attention Grad-CAM (referenced '{disease_en}').")
            elif disease_en in model.pathologies:
                target_index = model.pathologies.index(disease_en)
                display_disease = disease_en
            elif disease_trimmed in model.pathologies:
                target_index = model.pathologies.index(disease_trimmed)
                display_disease = disease_trimmed
            else:
                raise ValueError(
                    f"Invalid pathology name '{disease}'. "
                    f"Available pathologies: {model.pathologies}"
                )

        target_logit = logits[0, target_index]

        # 3. Compute gradients of target logit with respect to feature maps
        grads = torch.autograd.grad(target_logit, features, retain_graph=False)[0]

        # 4. Global average pooling of gradients to get channel importance weights
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # 5. Weighted combination of activation channels
        cam = torch.sum(weights * out_relu, dim=1, keepdim=True)

        # 6. Apply positive rectification or relative attention fallback
        if cam.max() > 0:
            cam_active = F.relu(cam)
        else:
            cam_active = cam - cam.min()

        # 7. Normalize heatmap to [0, 1] range
        cam_np = cam_active.detach().cpu().numpy()[0, 0]
        cam_min, cam_max = cam_np.min(), cam_np.max()

        if cam_max > cam_min:
            cam_normalized = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam_normalized = np.zeros_like(cam_np)

        # 8. Resize heatmap to original X-ray image dimensions (H, W)
        orig_h, orig_w = orig_gray.shape[:2]
        heatmap_resized = cv2.resize(
            cam_normalized,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR
        )

        # 9. Apply color map (JET)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # 10. Convert original grayscale image to 3-channel BGR
        orig_bgr = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)

        # 11. Overlay heatmap onto original X-ray (60% original image + 40% heatmap)
        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_bgr, 0.4, 0)

        # 12. Encode overlay image to PNG bytes
        png_bytes = encode_array_to_png(overlay)
        return png_bytes, display_disease

    except Exception as e:
        logger.error(f"Grad-CAM generation failed for disease '{disease}': {e}", exc_info=True)
        raise
