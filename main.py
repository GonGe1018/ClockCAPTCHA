from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import uuid
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import jwt
from dotenv import load_dotenv
from pipeline import generate_clock_captcha

load_dotenv()

app = FastAPI(title="Clock CAPTCHA Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=os.getenv("CORS_CREDENTIALS", "true").lower() == "true",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
CAPTCHA_EXPIRY_MINUTES = int(os.getenv("CAPTCHA_EXPIRY_MINUTES"))
GENERATED_IMAGES_DIR = os.getenv("GENERATED_IMAGES_DIR")

# redis로 바꾸기
active_sessions: Dict[str, dict] = {}
image_metadata: Dict[str, dict] = {}


def load_image_metadata():
    """생성된 이미지의 메타데이터 로드"""
    global image_metadata
    metadata_file = Path("image_metadata.json")

    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            image_metadata = json.load(f)
    else:
        image_metadata = {}
        images_dir = Path(GENERATED_IMAGES_DIR)
        if images_dir.exists():
            for img_file in images_dir.glob("*.png"):
                time_info = extract_time_from_filename(img_file.name)
                if time_info:
                    hour, minute = time_info
                    image_metadata[img_file.name] = {
                        "correct_time": f"{hour:02d}:{minute:02d}",
                        "hour": hour,
                        "minute": minute,
                        "filepath": str(img_file),
                    }

        with open(metadata_file, "w") as f:
            json.dump(image_metadata, f, indent=2)


def extract_time_from_filename(filename: str) -> Optional[tuple]:
    # *_clock_HH_MM_* 또는 clock_HH_MM
    match = re.search(r"clock_(\d{2})_(\d{2})", filename)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        return (hour, minute)
    return None


def generate_wrong_answers(correct_hour: int, correct_minute: int) -> List[str]:
    wrong_answers = []

    while len(wrong_answers) < 2:
        # 시간을 ±1-3시간, 분을 ±15-45분 범위에서 변경
        hour_offset = random.choice([-3, -2, -1, 1, 2, 3])
        minute_offset = random.choice([-45, -30, -15, 15, 30, 45])

        new_hour = (correct_hour + hour_offset) % 24
        new_minute = (correct_minute + minute_offset) % 60

        wrong_time = f"{new_hour:02d}:{new_minute:02d}"
        correct_time = f"{correct_hour:02d}:{correct_minute:02d}"

        # 정답과 다르고, 이미 생성된 오답과도 다른지 확인
        if wrong_time != correct_time and wrong_time not in wrong_answers:
            wrong_answers.append(wrong_time)

    return wrong_answers


def create_jwt_token(session_id: str) -> str:
    """검증 토큰 생성"""
    payload = {
        "session_id": session_id,
        "exp": datetime.utcnow() + timedelta(minutes=CAPTCHA_EXPIRY_MINUTES),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def verify_jwt_token(token: str) -> Optional[str]:
    """토큰 검증"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get("session_id")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


@app.on_event("startup")
async def startup_event():
    load_image_metadata()

    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path(GENERATED_IMAGES_DIR).mkdir(exist_ok=True)


@app.get("/")
async def root():
    return {"message": "Clock CAPTCHA Service", "version": "1.0.0"}


@app.get("/api.js")
async def get_api_script():
    api_js_path = Path("static/api.js")
    if api_js_path.exists():
        return FileResponse(api_js_path, media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="API script not found")


@app.get("/widget")
async def get_captcha_widget(request: Request, sitekey: str = "demo"):
    return templates.TemplateResponse(
        "captcha_widget.html", {"request": request, "sitekey": sitekey}
    )


@app.get("/api/challenge")
async def get_challenge():
    if not image_metadata:
        raise HTTPException(status_code=500, detail="No captcha images available")

    # 랜덤 이미지 선택
    image_filename = random.choice(list(image_metadata.keys()))
    image_info = image_metadata[image_filename]

    correct_time = image_info["correct_time"]
    correct_hour = image_info["hour"]
    correct_minute = image_info["minute"]

    # 오답 생성
    wrong_answers = generate_wrong_answers(correct_hour, correct_minute)

    # 선택지 섞기
    choices = [correct_time] + wrong_answers
    random.shuffle(choices)

    # 세션 생성
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "image_filename": image_filename,
        "correct_answer": correct_time,
        "choices": choices,
        "created_at": datetime.now(),
        "verified": False,
    }

    return JSONResponse(
        {
            "session_id": session_id,
            "image_url": f"/api/image/{session_id}",
            "choices": choices,
            "expires_in": CAPTCHA_EXPIRY_MINUTES * 60,
        }
    )


@app.get("/api/image/{session_id}")
async def get_captcha_image(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    image_filename = session["image_filename"]
    image_path = Path(GENERATED_IMAGES_DIR) / image_filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)


@app.post("/api/verify")
async def verify_answer(session_id: str = Form(...), answer: str = Form(...)):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]

    # 세션 만료 확인
    if datetime.now() - session["created_at"] > timedelta(
        minutes=CAPTCHA_EXPIRY_MINUTES
    ):
        del active_sessions[session_id]
        raise HTTPException(status_code=410, detail="Session expired")

    if answer == session["correct_answer"]:
        session["verified"] = True
        token = create_jwt_token(session_id)

        return JSONResponse(
            {"success": True, "token": token, "message": "Verification successful"}
        )
    else:
        return JSONResponse({"success": False, "message": "Incorrect answer"})


@app.post("/api/siteverify")
async def site_verify(token: str = Form(...), remoteip: Optional[str] = Form(None)):

    session_id = verify_jwt_token(token)

    if not session_id:
        return JSONResponse(
            {"success": False, "error-codes": ["invalid-input-response"]}
        )

    if session_id not in active_sessions:
        return JSONResponse({"success": False, "error-codes": ["timeout-or-duplicate"]})

    session = active_sessions[session_id]

    if not session.get("verified", False):
        return JSONResponse(
            {"success": False, "error-codes": ["invalid-input-response"]}
        )

    # 검증된 세션 삭제 (일회용)
    del active_sessions[session_id]

    return JSONResponse(
        {
            "success": True,
            "challenge_ts": session["created_at"].isoformat(),
            "hostname": "localhost",  # 실제 환경에서는 request의 호스트 사용
        }
    )


@app.get("/demo")
async def demo_page(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    if not os.path.exists(GENERATED_IMAGES_DIR) or not os.listdir(GENERATED_IMAGES_DIR):
        print("Generating captcha images...")
        generate_clock_captcha(
            all_clocks=True,
            add_noise=True,
            noise_type="mixed",
            noise_intensity=float(os.getenv("NOISE_INTENSITY")),
            output_size=(
                int(os.getenv("OUTPUT_SIZE_WIDTH")),
                int(os.getenv("OUTPUT_SIZE_HEIGHT")),
            ),
            grayscale=os.getenv("GRAYSCALE", "false").lower() == "true",
        )
        print("Images generated!")
        load_image_metadata()

    uvicorn.run(app, host=os.getenv("HOST"), port=int(os.getenv("PORT")))
