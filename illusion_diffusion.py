import torch
from PIL import Image
import random
from diffusers import (
    AutoencoderKL,
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


class IllusionDiffusion:
    """
    Illusion Diffusion HQ - Generate stunning high quality illusion artwork with Stable Diffusion
    Based on: https://huggingface.co/spaces/AP123/IllusionDiffusion
    """

    def __init__(
        self,
        device: str,
        base_model: str = "SG161222/Realistic_Vision_V5.1_noVAE",
    ):
        """
        Initialize the IllusionDiffusion pipeline

        Args:
            device: Device to run on ("mps", "cuda", "cpu")
            base_model: Base model to use for generation
        """
        self.device = device
        self.base_model = base_model

        self.sampler_map = {
            "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(
                config, use_karras=True, algorithm_type="sde-dpmsolver++"
            ),
            "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
        }

        self._load_models()

    def _load_models(self):
        """Load all required models and pipelines"""
        print("Loading models...")

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
        )

        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16
        )

        # Load safety checker
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.device)

        # Load feature extractor
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Load main pipeline
        self.main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=self.controlnet,
            vae=self.vae,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            torch_dtype=torch.float16,
        ).to(self.device)

        # Load image-to-image pipeline
        self.image_pipe = StableDiffusionControlNetImg2ImgPipeline(
            **self.main_pipe.components
        )

        print("Models loaded successfully!")

    @staticmethod
    def center_crop_resize(
        img: Image.Image, output_size: tuple = (512, 512)
    ) -> Image.Image:
        """Center crop and resize image"""
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

    @staticmethod
    def common_upscale(
        samples: torch.Tensor,
        width: int,
        height: int,
        upscale_method: str,
        crop: bool = False,
    ) -> torch.Tensor:
        """Common upscaling function"""
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

        return torch.nn.functional.interpolate(
            s, size=(height, width), mode=upscale_method
        )

    def upscale(
        self, samples: dict, upscale_method: str, scale_by: float
    ) -> torch.Tensor:
        """Upscale latent samples"""
        width = round(samples["images"].shape[3] * scale_by)
        height = round(samples["images"].shape[2] * scale_by)
        s = self.common_upscale(
            samples["images"], width, height, upscale_method, "disabled"
        )
        return s

    @staticmethod
    def convert_to_pil(image_path: str) -> Image.Image:
        """Convert image path to PIL Image"""
        return Image.open(image_path)

    def generate_image(
        self,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str = "low quality, blurry",
        guidance_scale: float = 8.0,
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        upscaler_strength: float = 0.5,
        seed: int = -1,
        sampler: str = "DPM++ Karras SDE",
        enable_upscaling: bool = True,
        num_inference_steps_base: int = 20,
        num_inference_steps_upscale: int = 20,
    ) -> Image.Image:
        """
        Generate illusion image

        Args:
            control_image: Control image for the illusion pattern
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale for generation
            controlnet_conditioning_scale: ControlNet conditioning scale
            control_guidance_start: Start of control guidance
            control_guidance_end: End of control guidance
            upscaler_strength: Strength of the upscaler
            seed: Random seed (-1 for random)
            sampler: Sampler to use
            enable_upscaling: Whether to enable upscaling
            num_inference_steps_base: Number of inference steps for base generation
            num_inference_steps_upscale: Number of inference steps for upscaling

        Returns:
            Generated PIL Image
        """
        # Prepare control images
        control_image_small = self.center_crop_resize(control_image)

        # Set scheduler
        self.main_pipe.scheduler = self.sampler_map[sampler](
            self.main_pipe.scheduler.config
        )

        # Set seed
        my_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
        generator = torch.Generator(device=self.device).manual_seed(my_seed)

        # Generate base image
        out = self.main_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image_small,
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=generator,
            control_guidance_start=float(control_guidance_start),
            control_guidance_end=float(control_guidance_end),
            num_inference_steps=num_inference_steps_base,
            output_type="latent" if enable_upscaling else "pil",
        )

        if not enable_upscaling:
            return out["images"][0]

        # Upscale if enabled
        control_image_large = self.center_crop_resize(control_image, (1024, 1024))
        upscaled_latents = self.upscale(out, "nearest-exact", 2)

        out_image = self.image_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=control_image_large,
            image=upscaled_latents,
            guidance_scale=float(guidance_scale),
            generator=generator,
            num_inference_steps=num_inference_steps_upscale,
            strength=upscaler_strength,
            control_guidance_start=float(control_guidance_start),
            control_guidance_end=float(control_guidance_end),
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        )

        return out_image["images"][0]
