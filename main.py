from illusion_diffusion import IllusionDiffusion

if __name__ == "__main__":
    illusion = IllusionDiffusion(device="mps")

    input_image = IllusionDiffusion.convert_to_pil("1566209829_4420.png")

    generated_image = illusion.generate_image(
        control_image=input_image,
        prompt="river, trees",
        negative_prompt="low quality",
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.4,
        upscaler_strength=0.75,
        seed=-1,
        sampler="Euler",
        enable_upscaling=False,
    )
    generated_image.show()
