import cv2
import numpy as np
import os
import random
import math
from PIL import Image


def draw_border_shape(img, center, radius, shape_type):
    """다양한 모양의 테두리를 그리는 함수"""

    if shape_type == "circle":
        cv2.circle(img, center, radius, (0, 0, 0), 4)

    elif shape_type == "square":
        # 정사각형
        half_size = int(radius * 0.8)
        top_left = (center[0] - half_size, center[1] - half_size)
        bottom_right = (center[0] + half_size, center[1] + half_size)
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 4)

    elif shape_type == "pentagon":
        # 오각형
        points = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 - math.pi / 2  # -90도에서 시작
            x = int(center[0] + radius * 0.9 * math.cos(angle))
            y = int(center[1] + radius * 0.9 * math.sin(angle))
            points.append([x, y])
        cv2.polylines(img, [np.array(points)], True, (0, 0, 0), 4)

    elif shape_type == "hexagon":
        # 육각형
        points = []
        for i in range(6):
            angle = i * 2 * math.pi / 6
            x = int(center[0] + radius * 0.9 * math.cos(angle))
            y = int(center[1] + radius * 0.9 * math.sin(angle))
            points.append([x, y])
        cv2.polylines(img, [np.array(points)], True, (0, 0, 0), 4)

    elif shape_type == "octagon":
        # 팔각형
        points = []
        for i in range(8):
            angle = i * 2 * math.pi / 8
            x = int(center[0] + radius * 0.9 * math.cos(angle))
            y = int(center[1] + radius * 0.9 * math.sin(angle))
            points.append([x, y])
        cv2.polylines(img, [np.array(points)], True, (0, 0, 0), 4)


def add_noise_pattern(img, center, radius, shape_type):
    """시계 테두리 바깥에 복잡한 노이즈 패턴 추가"""
    height, width = img.shape[:2]

    def is_inside_shape(x, y, center, radius, shape_type):
        """점이 시계 모양 내부에 있는지 확인"""
        if shape_type == "circle":
            dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            return dist < radius + 10

        elif shape_type == "square":
            half_size = int(radius * 0.8) + 10
            return abs(x - center[0]) < half_size and abs(y - center[1]) < half_size

        elif shape_type == "pentagon":
            # 오각형 내부 체크 (간단한 원형 근사)
            dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            return dist < radius * 0.9 + 10

        elif shape_type == "hexagon":
            # 육각형 내부 체크 (간단한 원형 근사)
            dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            return dist < radius * 0.9 + 10

        elif shape_type == "octagon":
            # 팔각형 내부 체크 (간단한 원형 근사)
            dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            return dist < radius * 0.9 + 10

        return False

    # 1. 더 많은 랜덤 점들 (스프링클)
    for _ in range(random.randint(300, 500)):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        if not is_inside_shape(x, y, center, radius, shape_type):
            cv2.circle(img, (x, y), random.randint(1, 3), (0, 0, 0), -1)

    # 2. 더 많은 랜덤 선들 (크로스해치)
    for _ in range(random.randint(50, 80)):
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        x2 = x1 + random.randint(-40, 40)
        y2 = y1 + random.randint(-40, 40)
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        # 선이 시계 내부를 지나가지 않도록 체크
        line_crosses_clock = False
        for t in np.linspace(0, 1, 30):
            lx = int(x1 + t * (x2 - x1))
            ly = int(y1 + t * (y2 - y1))
            if is_inside_shape(lx, ly, center, radius, shape_type):
                line_crosses_clock = True
                break

        if not line_crosses_clock:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), random.randint(1, 3))

    # 3. 더 많은 기하학적 모양들
    for _ in range(random.randint(30, 50)):
        x = random.randint(10, width - 10)
        y = random.randint(10, height - 10)

        if not is_inside_shape(x, y, center, radius, shape_type):
            shape_type_noise = random.choice(
                ["circle", "square", "triangle", "diamond", "cross"]
            )
            size = random.randint(3, 12)

            if shape_type_noise == "circle":
                cv2.circle(img, (x, y), size, (0, 0, 0), random.randint(1, 2))
            elif shape_type_noise == "square":
                cv2.rectangle(
                    img,
                    (x - size, y - size),
                    (x + size, y + size),
                    (0, 0, 0),
                    random.randint(1, 2),
                )
            elif shape_type_noise == "triangle":
                pts = np.array(
                    [[x, y - size], [x - size, y + size], [x + size, y + size]],
                    np.int32,
                )
                cv2.polylines(img, [pts], True, (0, 0, 0), random.randint(1, 2))
            elif shape_type_noise == "diamond":
                pts = np.array(
                    [[x, y - size], [x + size, y], [x, y + size], [x - size, y]],
                    np.int32,
                )
                cv2.polylines(img, [pts], True, (0, 0, 0), random.randint(1, 2))
            elif shape_type_noise == "cross":
                cv2.line(
                    img, (x - size, y), (x + size, y), (0, 0, 0), random.randint(1, 2)
                )
                cv2.line(
                    img, (x, y - size), (x, y + size), (0, 0, 0), random.randint(1, 2)
                )

    # 4. 더 조밀한 텍스처 패턴 (격자)
    grid_spacing = random.randint(5, 10)
    for x in range(0, width, grid_spacing):
        for y in range(0, height, grid_spacing):
            if (
                not is_inside_shape(x, y, center, radius, shape_type)
                and random.random() < 0.4
            ):
                cv2.circle(img, (x, y), 1, (0, 0, 0), -1)

    # 5. 복잡한 곡선 패턴
    for _ in range(random.randint(10, 20)):
        points = []
        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)

        for i in range(random.randint(10, 30)):
            if i == 0:
                points.append([start_x, start_y])
            else:
                prev_x, prev_y = points[-1]
                new_x = prev_x + random.randint(-15, 15)
                new_y = prev_y + random.randint(-15, 15)
                new_x = max(0, min(width - 1, new_x))
                new_y = max(0, min(height - 1, new_y))

                if not is_inside_shape(new_x, new_y, center, radius, shape_type):
                    points.append([new_x, new_y])

        if len(points) > 3:
            pts = np.array(points, np.int32)
            cv2.polylines(img, [pts], False, (0, 0, 0), random.randint(1, 2))

    # 6. 랜덤 각도의 선들 (더 복잡한 해칭)
    for _ in range(random.randint(20, 40)):
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(20, 60)

        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)
        end_x = int(start_x + length * math.cos(angle))
        end_y = int(start_y + length * math.sin(angle))

        end_x = max(0, min(width - 1, end_x))
        end_y = max(0, min(height - 1, end_y))

        # 선이 시계를 지나가지 않는지 체크
        line_valid = True
        for t in np.linspace(0, 1, 20):
            lx = int(start_x + t * (end_x - start_x))
            ly = int(start_y + t * (end_y - start_y))
            if is_inside_shape(lx, ly, center, radius, shape_type):
                line_valid = False
                break

        if line_valid:
            cv2.line(
                img, (start_x, start_y), (end_x, end_y), (0, 0, 0), random.randint(1, 2)
            )

    # 7. 작은 원들의 클러스터
    for _ in range(random.randint(5, 15)):
        cluster_x = random.randint(20, width - 20)
        cluster_y = random.randint(20, height - 20)

        if not is_inside_shape(cluster_x, cluster_y, center, radius, shape_type):
            for _ in range(random.randint(5, 15)):
                offset_x = random.randint(-15, 15)
                offset_y = random.randint(-15, 15)
                x = cluster_x + offset_x
                y = cluster_y + offset_y

                if (
                    0 <= x < width
                    and 0 <= y < height
                    and not is_inside_shape(x, y, center, radius, shape_type)
                ):
                    cv2.circle(img, (x, y), random.randint(1, 3), (0, 0, 0), -1)


def get_number_position(center, radius, hour_index, shape_type):
    """각 모양에 맞는 숫자 위치를 계산하는 함수"""
    angle = hour_index * 30  # 30도씩
    angle_rad = np.deg2rad(angle)

    if shape_type == "circle":
        # 원형: 기본 반지름에서 안쪽으로
        text_radius = radius - 25

    elif shape_type == "square":
        # 정사각형: 모서리에 맞춰 위치 조정
        text_radius = radius * 0.6

    elif shape_type == "pentagon":
        # 오각형: 꼭짓점 근처에 배치
        text_radius = radius * 0.65

    elif shape_type == "hexagon":
        # 육각형: 꼭짓점과 변 사이에 배치
        text_radius = radius * 0.7

    elif shape_type == "octagon":
        # 팔각형: 변에 맞춰 배치
        text_radius = radius * 0.7

    else:
        text_radius = radius - 25

    x = int(center[0] + text_radius * np.sin(angle_rad))
    y = int(center[1] - text_radius * np.cos(angle_rad))

    return x, y


def draw_clock(
    hour,
    minute,
    size=256,
    save_dir="clocks",
    min_angle_diff=10,
    hour_thickness=4,
    minute_thickness=4,
    number_thickness=0,
    number_font=0.7,
):
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

    # 겹치거나 가까운 경우 제외 (기존)
    if angle_diff <= min_angle_diff:
        return  # 겹치거나 가까우면 저장하지 않음

    # 일직선이 되는 경우 제외 (180도 ± 10도 범위)
    if abs(angle_diff - 180) <= min_angle_diff:
        return  # 일직선이면 저장하지 않음

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    center = (size // 2, size // 2)
    radius = size // 2 - 10

    # 랜덤하게 테두리 모양 선택
    shapes = ["circle", "square", "pentagon", "hexagon", "octagon"]
    selected_shape = random.choice(shapes)

    # 다양한 모양의 시계 테두리
    draw_border_shape(img, center, radius, selected_shape)

    # 숫자 (1-12)
    if number_thickness > 0:
        for i in range(12):
            # 각 모양에 맞는 숫자 위치 계산
            text_x, text_y = get_number_position(center, radius, i, selected_shape)

            # 숫자 (12시 위치가 0이므로 12부터 시작)
            number = 12 if i == 0 else i

            # 텍스트 크기 계산해서 중앙 정렬
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(
                str(number), font, number_font, number_thickness
            )[0]
            text_x -= text_size[0] // 2
            text_y += text_size[1] // 2

            cv2.putText(
                img,
                str(number),
                (text_x, text_y),
                font,
                number_font,
                (0, 0, 0),
                number_thickness,
            )

    # 분침
    minute_angle = np.deg2rad(minute_angle_deg)
    min_length = int(radius * 0.85)
    min_x = int(center[0] + min_length * np.sin(minute_angle))
    min_y = int(center[1] - min_length * np.cos(minute_angle))
    cv2.line(img, center, (min_x, min_y), (0, 0, 0), minute_thickness)

    # 시침
    hour_angle = np.deg2rad(hour_angle_deg)
    hour_length = int(radius * 0.55)
    hour_x = int(center[0] + hour_length * np.sin(hour_angle))
    hour_y = int(center[1] - hour_length * np.cos(hour_angle))
    cv2.line(img, center, (hour_x, hour_y), (0, 0, 0), hour_thickness)

    # 중심점
    cv2.circle(img, center, 8, (0, 0, 0), -1)

    # 복잡한 노이즈 패턴 추가 (시계 테두리 바깥, 모양별로 정확히)
    add_noise_pattern(img, center, radius, selected_shape)

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


def get_all_clock_images():
    save_dir = "clocks"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        generate_all_clocks()
    clock_images = [f for f in os.listdir(save_dir) if f.endswith(".png")]
    if not clock_images:
        generate_all_clocks()
        clock_images = [f for f in os.listdir(save_dir) if f.endswith(".png")]
    return [(Image.open(os.path.join(save_dir, img)), img) for img in clock_images]


if __name__ == "__main__":
    input("Press Enter to generate clock images...")
    generate_all_clocks()
