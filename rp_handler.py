# import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict, List
from PIL import Image, ImageFilter

from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel
from image_gen_aux import DepthPreprocessor

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

from colors import ade_palette
from utils import map_colors_rgb

# --------------------------- КОНСТАНТЫ ----------------------------------- #
MAX_SEED = np.iinfo(np.int32).max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_STEPS = 250

TARGET_RES = 1024

DEFAULT_CONTROL_ITEMS = [
    "windowpane;window",
    "column;pillar",
    # "door;double;door",
]

logger = RunPodLogger()


# ------------------------- ФУНКЦИИ-ПОМОЩНИКИ ----------------------------- #
def filter_items(colors_list, items_list, items_to_remove):
    keep_c, keep_i = [], []
    for c, it in zip(colors_list, items_list):
        if it not in items_to_remove:
            keep_c.append(c)
            keep_i.append(it)
    return keep_c, keep_i


def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def round_to_multiple(x, m=8):
    return (x // m) * m


def compute_work_resolution(w, h, max_side=1024):
    # масштабируем так, чтобы большая сторона <= max_side
    scale = min(max_side / max(w, h), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # выравниваем до кратных 8
    new_w = round_to_multiple(new_w, 8)
    new_h = round_to_multiple(new_h, 8)
    return max(new_w, 8), max(new_h, 8)


def normalize_control_items(raw) -> List[str]:
    """
    Приводит control_items к списку строк.
    Поддерживает: None, строку с запятыми/новыми строками, список.
    """
    if raw is None:
        return DEFAULT_CONTROL_ITEMS[:]

    if isinstance(raw, str):
        # разделяем по запятым или переносам
        parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()] ## noqa
        return parts or DEFAULT_CONTROL_ITEMS[:]

    if isinstance(raw, (list, tuple)):
        cleaned = [str(x).strip() for x in raw if str(x).strip()]
        return cleaned or DEFAULT_CONTROL_ITEMS[:]

    # на всякий случай
    return DEFAULT_CONTROL_ITEMS[:]


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #
# БАЗА: FLUX.1-dev + depth ControlNet
base_repo = "black-forest-labs/FLUX.1-dev"
controlnet_model = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"

CONTROLNET = FluxControlNetModel.from_pretrained(
    controlnet_model,
    torch_dtype=DTYPE
)

PIPELINE = FluxControlNetInpaintPipeline.from_pretrained(
    base_repo,
    controlnet=CONTROLNET,
    torch_dtype=DTYPE
).to(DEVICE)

processor = DepthPreprocessor.from_pretrained(
    "LiheYoung/depth-anything-large-hf"
)

seg_image_processor = AutoImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)
image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)


@torch.inference_mode()
@torch.autocast(DEVICE)
def segment_image(image):
    """
    Segments an image using a semantic segmentation model.

    Args:
        image (PIL.Image): The input image to be segmented.
        image_processor (AutoImageProcessor): The processor to prepare the
            image for segmentation.
        image_segmentor (SegformerForSemanticSegmentation): The semantic
            segmentation model used to identify different segments in image.

    Returns:
        Image: The segmented image with each segment colored differently based
            on its identified class.
    """
    pixel_values = seg_image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = seg_image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)
    seg_image = Image.fromarray(color_seg).convert("RGB")

    return seg_image


# ------------------------- ОСНОВНОЙ HANDLER ------------------------------ #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = job.get("input", {})
        image_url = payload.get("image_url")
        mask_url = payload.get("mask_url")
        if not image_url:
            return {"error": "'image_url' is required"}
        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        # ⚙️ Новые параметры
        # neg = payload.get("negative_prompt") or \
        #     "blurry, low quality, watermark, logo, text, artifacts, distorted geometry, misaligned perspective"

        img_strength = float(payload.get(
            "img_strength", 0.5))
        # true_cfg_scale = float(payload.get("true_cfg_scale", 3.0))
        cn_scale = float(payload.get("controlnet_conditioning_scale", 0.5))
        guidance_scale = float(payload.get("guidance_scale", 3.5))
        steps = min(int(payload.get("steps", MAX_STEPS)), MAX_STEPS)

        control_guidance_start = float(payload.get(
            "control_guidance_start", 0.0))
        control_guidance_end = float(payload.get(
            "control_guidance_end", 0.8))

        seed = int(payload.get("seed", random.randint(0, MAX_SEED)))
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # mask
        control_items = normalize_control_items(payload.get("control_items"))
        mask_blur_radius = float(payload.get("mask_blur_radius", 3))
        image_blur_radius = float(payload.get("image_blur_radius", 1))

        image_pil = url_to_pil(image_url)

        orig_w, orig_h = image_pil.size
        work_w, work_h = compute_work_resolution(orig_w, orig_h, TARGET_RES)
        image_pil = image_pil.resize((work_w, work_h),
                                     Image.Resampling.LANCZOS)

        # canny-карта
        control_image = processor(image_pil)[0].convert("RGB")

        # РАБОТА С МАСКОЙ
        if not mask_url:
            real_seg = np.array(
                segment_image(image_pil)
            )
            unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]),
                                      axis=0)
            unique_colors = [tuple(color) for color in unique_colors]
            segment_items = [map_colors_rgb(i) for i in unique_colors]

            chosen_colors, segment_items = filter_items(
                colors_list=unique_colors,
                items_list=segment_items,
                items_to_remove=control_items,
            )

            logger.log(f"SEGMENTED ITEMS {segment_items}")

            # Установим маску
            mask = np.zeros_like(real_seg)
            for color in chosen_colors:
                color_matches = (real_seg == color).all(axis=2)
                mask[color_matches] = 1
            mask_image = Image.fromarray(
                (mask * 255).astype(np.uint8)).convert("RGB")
            mask_image = mask_image.filter(
                ImageFilter.GaussianBlur(radius=image_blur_radius))
        else:
            mask_image = url_to_pil(mask_url)
            mask_image = mask_image.resize((work_w, work_h),
                                           Image.Resampling.LANCZOS)

        # ------------------ генерация ---------------- #
        images = PIPELINE(
            prompt=prompt,
            # negative_prompt=neg,
            # true_cfg_scale=true_cfg_scale,
            image=image_pil,
            mask_image=mask_image,
            control_image=control_image,
            controlnet_conditioning_scale=cn_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            num_inference_steps=steps,
            strength=img_strength,
            guidance_scale=guidance_scale,
            generator=generator,
            width=work_w,
            height=work_h,
            mask_blur_radius=mask_blur_radius
        ).images

        return {
            "images_base64": [pil_to_b64(i) for i in images],
            "time": round(time.time() - job["created"], 2) if "created" in job else None,
            "steps": steps,
            "seed": seed
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps' или размер изображения."}
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
