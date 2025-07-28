from illusion_diffusion import IllusionDiffusion
from clock_dataset import get_random_clock_image

from PIL import Image

illusion = IllusionDiffusion(device="mps")


def generate_clock_captcha(
    prompt: str = "ancient artifact, runes, stone dial",
) -> Image.Image:
    clock_image = get_random_clock_image()
    generated_image = illusion.generate_image(
        control_image=clock_image,
        prompt=prompt + ", simple background",
        negative_prompt="low quality, blurry, deformed clock",
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.4,
        upscaler_strength=0.75,
        seed=-1,
        sampler="Euler",
        enable_upscaling=False,
        num_inference_steps_base=15,
        num_inference_steps_upscale=20,
    )

    generated_image.show()

    return generated_image


if __name__ == "__main__":
    prompts = [
        "ancient artifact, runes, stone dial",
        "sci-fi chronometer, glowing dials, clean futuristic design",
        "cyberpunk interface, neon grid, tech pattern",
    ]

    generated_captcha = generate_clock_captcha(prompt=prompts[3])
    generated_captcha.save("generated_clock_captcha.png")
    print("Clock CAPTCHA generated and saved as 'generated_clock_captcha.png'.")
