from pipeline import generate_clock_captcha


if __name__ == "__main__":
    generated_captcha = generate_clock_captcha(showImage=True)
    generated_captcha.save("generated_clock_captcha.png")
