import os
import torch

from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel
from controlnet_aux import CannyDetector

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ------------------------- пайплайн -------------------------
def get_pipeline():
    base_repo = "black-forest-labs/FLUX.1-dev"
    controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Canny'

    CONTROLNET = FluxControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=torch.bfloat16
    )

    FluxControlNetInpaintPipeline.from_pretrained(
        base_repo,
        controlnet=CONTROLNET,
        torch_dtype=torch.bfloat16
    )

    CannyDetector()

    AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )


if __name__ == "__main__":
    get_pipeline()
