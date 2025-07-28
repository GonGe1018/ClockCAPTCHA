import cv2
import numpy as np
import os
from PIL import Image


def draw_clock(hour, minute, size=256, save_dir="clocks", min_angle_diff=10):
    """
    hour: 0~11
    minute: 0~55 (5분 단위)
    size: 이미지 크기 (정사각형)
    save_dir: 저장 폴더
    min_angle_diff: 시침과 분침의 최소 각도 차이(도)
    """
    # 시침, 분침 각도 계산
    minute_angle_deg = minute * 6
    hour_angle_deg = (hour % 12) * 30 + minute * 0.5
    angle_diff = abs(minute_angle_deg - hour_angle_deg)
    angle_diff = min(angle_diff, 360 - angle_diff)
    if angle_diff <= min_angle_diff:
        return  # 겹치거나 가까우면 저장하지 않음

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    center = (size // 2, size // 2)
    radius = size // 2 - 10

    # 시계 테두리
    cv2.circle(img, center, radius, (0, 0, 0), 4)

    # 눈금 (12개)
    for i in range(12):
        angle = np.deg2rad(i * 30)
        x1 = int(center[0] + (radius - 10) * np.sin(angle))
        y1 = int(center[1] - (radius - 10) * np.cos(angle))
        x2 = int(center[0] + radius * np.sin(angle))
        y2 = int(center[1] - radius * np.cos(angle))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # 분침
    minute_angle = np.deg2rad(minute_angle_deg)
    min_length = int(radius * 0.85)
    min_x = int(center[0] + min_length * np.sin(minute_angle))
    min_y = int(center[1] - min_length * np.cos(minute_angle))
    cv2.line(img, center, (min_x, min_y), (0, 0, 0), 6)

    # 시침
    hour_angle = np.deg2rad(hour_angle_deg)
    hour_length = int(radius * 0.55)
    hour_x = int(center[0] + hour_length * np.sin(hour_angle))
    hour_y = int(center[1] - hour_length * np.cos(hour_angle))
    cv2.line(img, center, (hour_x, hour_y), (0, 0, 0), 10)

    # 중심점
    cv2.circle(img, center, 8, (0, 0, 0), -1)

    # 파일명 예: clock_03_05.png
    filename = f"clock_{hour:02d}_{minute:02d}.png"
    cv2.imwrite(os.path.join(save_dir, filename), img)


def generate_all_clocks():
    for hour in range(12):
        for minute in range(0, 60, 5):
            draw_clock(hour, minute)


def get_random_clock_image():
    save_dir = "clocks"
    clock_images = [f for f in os.listdir(save_dir) if f.endswith(".png")]
    if not clock_images:
        generate_all_clocks()
        clock_images = [f for f in os.listdir(save_dir) if f.endswith(".png")]
    random_image = np.random.choice(clock_images)
    image_path = os.path.join(save_dir, random_image)
    return Image.open(image_path)


if __name__ == "__main__":
    input("Press Enter to generate clock images...")
    generate_all_clocks()
