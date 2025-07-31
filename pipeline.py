from illusion_diffusion import IllusionDiffusion
from clock_dataset import get_random_clock_image, get_all_clock_images
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import uuid
import numpy as np
from typing import List
import os


illusion = IllusionDiffusion(device=os.getenv("DEVICE"))


def add_adversarial_noise(
    image: Image.Image, noise_type: str = "mixed", intensity: float = 0.3
) -> Image.Image:
    """
    AI가 알아보기 힘든 다양한 노이즈를 이미지에 추가합니다.

    Args:
        image: 원본 PIL Image
        noise_type: 노이즈 타입 ("gaussian", "salt_pepper", "geometric", "mixed")
        intensity: 노이즈 강도 (0.0 ~ 1.0)

    Returns:
        PIL Image
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    if noise_type == "gaussian" or noise_type == "mixed":
        # 가우시안 노이즈 - AI 모델이 잘못 인식하도록
        gaussian_noise = np.random.normal(0, intensity * 25, img_array.shape).astype(
            np.int16
        )
        img_array = np.clip(img_array.astype(np.int16) + gaussian_noise, 0, 255).astype(
            np.uint8
        )

    if noise_type == "salt_pepper" or noise_type == "mixed":
        # Salt & Pepper 노이즈 - 무작위 픽셀 변경
        noise_mask = np.random.random(img_array.shape[:2]) < intensity * 0.05
        if len(img_array.shape) == 3:  # 컬러 이미지인 경우
            # 각 채널에 동일한 값 적용 (흑백 점)
            salt_pepper_values = np.random.choice([0, 255], size=np.sum(noise_mask))
            for channel in range(img_array.shape[2]):
                img_array[noise_mask, channel] = salt_pepper_values
        else:  # 그레이스케일 이미지인 경우
            img_array[noise_mask] = np.random.choice([0, 255], size=np.sum(noise_mask))

    noisy_image = Image.fromarray(img_array)

    if noise_type == "geometric" or noise_type == "mixed":
        # 기하학적 왜곡 - 미세한 회전과 왜곡
        angle = random.uniform(-intensity * 2, intensity * 2)
        noisy_image = noisy_image.rotate(angle, fillcolor="white")

        # 색상 왜곡
        enhancer = ImageEnhance.Color(noisy_image)
        noisy_image = enhancer.enhance(
            1 + random.uniform(-intensity * 0.3, intensity * 0.3)
        )

        # 대비 조정
        enhancer = ImageEnhance.Contrast(noisy_image)
        noisy_image = enhancer.enhance(
            1 + random.uniform(-intensity * 0.2, intensity * 0.2)
        )

    if noise_type == "mixed":
        # 추가적인 혼합 노이즈 - 랜덤 선과 점들
        draw = ImageDraw.Draw(noisy_image)
        for _ in range(int(intensity * 10)):
            # 무작위 선 그리기
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            color = random.randint(100, 200)
            draw.line([(x1, y1), (x2, y2)], fill=(color, color, color), width=1)

        for _ in range(int(intensity * 20)):
            # 무작위 점 그리기
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            color = random.randint(50, 255)
            draw.point([(x, y)], fill=(color, color, color))

    return noisy_image


def post_process_image(
    image: Image.Image,
    output_size: tuple = None,
    grayscale: bool = False,
    add_noise: bool = False,
    noise_type: str = "mixed",
    noise_intensity: float = 0.6,
) -> Image.Image:
    """
    이미지 후처리: 크기 조정, 노이즈 추가 및 흑백 변환

    Args:
        image: 원본 PIL Image
        output_size: (width, height) 튜플. None이면 크기 변경 안함
        grayscale: True이면 흑백으로 변환
        add_noise: True이면 adversarial 노이즈 추가
        noise_type: 노이즈 타입 ("gaussian", "salt_pepper", "geometric", "mixed")
        noise_intensity: 노이즈 강도 (0.0 ~ 1.0)

    Returns:
        후처리된 PIL Image
    """
    processed_image = image.copy()

    # 1. 크기 조정 (먼저)
    if output_size is not None:
        processed_image = processed_image.resize(output_size, Image.Resampling.LANCZOS)

    # 2. 노이즈 추가 (리사이즈 후)
    if add_noise:
        processed_image = add_adversarial_noise(
            processed_image, noise_type=noise_type, intensity=noise_intensity
        )

    # 3. 흑백 변환 (마지막)
    if grayscale:
        processed_image = processed_image.convert("L")
        # 흑백 이미지를 RGB로 다시 변환 (3채널 유지)
        processed_image = processed_image.convert("RGB")

    return processed_image


def generate_clock_captcha(
    prompt: str = None,
    all_clocks: bool = False,
    save_dir: str = "./generated_captchas",
    add_noise: bool = False,
    noise_type: str = "mixed",
    noise_intensity: float = 0.6,
    output_size: tuple = None,
    grayscale: bool = False,
) -> List[Image.Image]:
    """
    Generate a clock CAPTCHA image with a random clock face and a prompt.
    Args:
        prompt: Text prompt for the illusion generation
        all_clocks: If True, use all clock images instead of a random one
        save_dir: Directory to save generated images
        add_noise: If True, add adversarial noise to make it harder for AI to recognize
        noise_type: Type of noise to add ("gaussian", "salt_pepper", "geometric", "mixed")
        noise_intensity: Intensity of noise (0.0 to 1.0)
        output_size: Tuple (width, height) for output image size. If None, keeps original size
        grayscale: If True, convert output to grayscale
    Returns:
        PIL Image List
    """

    random_prompts = [
        "ancient artifact, runes, stone dial",
        "sci-fi chronometer, glowing dials, clean futuristic design",
    ]

    pid = str(uuid.uuid4())[:8]

    if all_clocks:
        clock_data = get_all_clock_images()

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        generated_images = []
        for clock_image, filename in clock_data:
            current_prompt = prompt if prompt else random.choice(random_prompts)
            generated_image = illusion.generate_image(
                control_image=clock_image,
                prompt=current_prompt + ", simple background",
                negative_prompt="low quality, blurry",
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.5,
                upscaler_strength=0.75,
                seed=-1,
                sampler="Euler",
                enable_upscaling=False,
                num_inference_steps_base=15,
                num_inference_steps_upscale=20,
            )

            generated_image = post_process_image(
                generated_image,
                output_size=output_size,
                grayscale=grayscale,
                add_noise=add_noise,
                noise_type=noise_type,
                noise_intensity=noise_intensity,
            )

            generated_images.append(generated_image)
            noise_suffix = f"_noise_{noise_type}_{noise_intensity}" if add_noise else ""
            size_suffix = f"_{output_size[0]}x{output_size[1]}" if output_size else ""
            grayscale_suffix = "_bw" if grayscale else ""
            output_filename = f"{pid}_captcha_{filename.replace('.png', '')}{noise_suffix}{size_suffix}{grayscale_suffix}.png"
            generated_image.save(Path(save_dir) / output_filename)

        return generated_images

    else:
        clock_image = get_random_clock_image()

        current_prompt = prompt if prompt else random.choice(random_prompts)
        generated_image = illusion.generate_image(
            control_image=clock_image,
            prompt=current_prompt + ", simple background",
            negative_prompt="low quality, blurry",
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.5,
            upscaler_strength=0.75,
            seed=-1,
            sampler="Euler",
            enable_upscaling=False,
            num_inference_steps_base=15,
            num_inference_steps_upscale=20,
        )

        generated_image = post_process_image(
            generated_image,
            output_size=output_size,
            grayscale=grayscale,
            add_noise=add_noise,
            noise_type=noise_type,
            noise_intensity=noise_intensity,
        )

        generated_image.show()

        return [generated_image]


if __name__ == "__main__":
    generated_captcha = generate_clock_captcha(
        all_clocks=True,
        add_noise=True,
        noise_type="mixed",
        noise_intensity=0.6,
        output_size=(128, 128),
        grayscale=False,
    )
