import os
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "hf_home"))

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from transformers import Dinov2Config, Dinov2Model


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_PATH = REPO_ROOT / "SAM.png"
CHECKPOINT_CANDIDATES = [
    REPO_ROOT / "dinov2_BEST_aug.pth",
    REPO_ROOT / "dinov2_BEST_unaug.pth",
    REPO_ROOT / "dinov2_base.pth",
]
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
MIN_MARGIN_CM = 1.5


class DINOv2ForHeightRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = Dinov2Model(
            Dinov2Config(
                image_size=518,
                patch_size=14,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
        )
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        prediction = self.regressor(cls_token)
        return prediction.squeeze(-1)


def resolve_checkpoints() -> list[Path]:
    available_checkpoints = [
        checkpoint_path for checkpoint_path in CHECKPOINT_CANDIDATES if checkpoint_path.exists()
    ]
    if available_checkpoints:
        return available_checkpoints

    raise FileNotFoundError(
        "No model checkpoint was found. Expected one of: "
        + ", ".join(path.name for path in CHECKPOINT_CANDIDATES)
    )


def centimeters_to_feet_inches(height_cm: float) -> tuple[int, int]:
    total_inches = round(height_cm / 2.54)
    feet = total_inches // 12
    inches = total_inches % 12
    return feet, inches


def load_image(uploaded_file) -> tuple[Image.Image, str]:
    if uploaded_file is None:
        if not DEFAULT_IMAGE_PATH.exists():
            raise FileNotFoundError(
                f"Default example image not found at {DEFAULT_IMAGE_PATH.relative_to(REPO_ROOT)}."
            )
        return Image.open(DEFAULT_IMAGE_PATH).convert("RGB"), "Using default example image: SAM.png"

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid JPG or PNG image.") from exc
    except OSError as exc:
        raise ValueError("The uploaded image could not be opened.") from exc

    return image, f"Using uploaded image: {uploaded_file.name}"


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return IMAGE_TRANSFORM(image).unsqueeze(0)


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[list[tuple[nn.Module, Path]], str]:
    checkpoint_paths = resolve_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_models: list[tuple[nn.Module, Path]] = []

    for checkpoint_path in checkpoint_paths:
        model = DINOv2ForHeightRegression()
        model.to(device)

        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model weights from {checkpoint_path.name}."
            ) from exc

        model.eval()
        loaded_models.append((model, checkpoint_path))

    return loaded_models, device


def predict_heights_cm(
    models: list[tuple[nn.Module, Path]], image: Image.Image, device: str
) -> list[float]:
    try:
        pixel_values = preprocess_image(image)
        pixel_values = pixel_values.to(device)
        predictions: list[float] = []

        with torch.no_grad():
            for model, _checkpoint_path in models:
                prediction = model(pixel_values)
                predictions.append(float(prediction.item()))

        return predictions
    except Exception as exc:
        raise RuntimeError("The model could not generate a prediction for this image.") from exc


def main() -> None:
    st.set_page_config(page_title="Height Prediction App", page_icon="📏")
    st.title("Height Prediction App")
    st.write(
        "Upload a photo to estimate height. The app automatically resizes and preprocesses "
        "the image internally, so any JPG or PNG image size should work."
    )

    uploaded_file = st.file_uploader(
        "Upload a photo",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    try:
        image, image_status = load_image(uploaded_file)
        st.image(image, caption=image_status, use_container_width=True)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    try:
        models, device = load_models()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except RuntimeError as exc:
        st.error(f"Model loading failure: {exc}")
        return

    device_label = "Running on GPU" if device == "cuda" else "Running on CPU"
    checkpoint_names = ", ".join(checkpoint_path.name for _, checkpoint_path in models)
    st.caption(f"{device_label} | Loaded checkpoints: {checkpoint_names}")

    if st.button("Predict Height", type="primary"):
        try:
            predictions = predict_heights_cm(models, image, device)
        except RuntimeError as exc:
            st.error(f"Inference failure: {exc}")
            return

        mean_prediction = float(np.mean(predictions))
        std_prediction = float(np.std(predictions))
        if len(predictions) == 1:
            margin_cm = MIN_MARGIN_CM
        else:
            margin_cm = max(MIN_MARGIN_CM, 2 * std_prediction)

        height_cm = mean_prediction
        feet, inches = centimeters_to_feet_inches(height_cm)
        lower_bound = round(height_cm - margin_cm)
        upper_bound = round(height_cm + margin_cm)
        lower_feet, lower_inches = centimeters_to_feet_inches(lower_bound)
        upper_feet, upper_inches = centimeters_to_feet_inches(upper_bound)

        st.subheader("Prediction")
        st.success(
            f"Predicted height: {round(height_cm)} cm ({feet} ft {inches} in)"
        )
        st.write(
            f"Approximate range: {lower_bound} cm ({lower_feet} ft {lower_inches} in) "
            f"to {upper_bound} cm ({upper_feet} ft {upper_inches} in)"
        )


if __name__ == "__main__":
    main()
