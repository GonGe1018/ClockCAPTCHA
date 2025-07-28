import torch
from PIL import Image
import random
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)


def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)
    left = (width - new_dimension) / 2
    top = (height - new_dimension) / 2
    right = (width + new_dimension) / 2
    bottom = (height + new_dimension) / 2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img


def common_upscale(samples, width, height, upscale_method, crop=False):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:, :, y : old_height - y, x : old_width - x]
    else:
        s = samples

    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)


def upscale(samples, upscale_method, scale_by):
    # s = samples.copy()
    width = round(samples["images"].shape[3] * scale_by)
    height = round(samples["images"].shape[2] * scale_by)
    s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
    return s


def convert_to_pil(base64_image):
    pil_image = Image.open(base64_image)
    return pil_image


DEVICE = "mps"

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"

SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(
        config, use_karras=True, algorithm_type="sde-dpmsolver++"
    ),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
)
controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16
)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker"
).to(DEVICE)
feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    vae=vae,
    safety_checker=safety_checker,
    feature_extractor=feature_extractor,
    torch_dtype=torch.float16,
).to(DEVICE)

image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)


def generate_image(
    control_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 8.0,
    controlnet_conditioning_scale: float = 1,
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 1.0,
    upscaler_strength: float = 0.5,
    seed: int = -1,
    sampler="DPM++ Karras SDE",
):
    control_image_small = center_crop_resize(control_image)
    control_image_large = center_crop_resize(control_image, (1024, 1024))

    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)
    my_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    generator = torch.Generator(device=DEVICE).manual_seed(my_seed)

    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        num_inference_steps=15,
        output_type="latent",
    )
    upscaled_latents = upscale(out, "nearest-exact", 2)
    out_image = image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image_large,
        image=upscaled_latents,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=20,
        strength=upscaler_strength,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
    )

    return out_image["images"][0]


if __name__ == "__main__":
    input_image = convert_to_pil("1566209829_4420.png")
    generated_image = generate_image(
        control_image=input_image,
        prompt="A beautiful landscape",
        negative_prompt="blurry, low quality",
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.4,
        upscaler_strength=0.75,
        seed=42,
        sampler="DPM++ Karras SDE",
    )
    generated_image.show()
