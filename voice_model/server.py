import os
import io
import argparse
import uvicorn
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from TTS.tts.models.vits import Vits
from TTS.tts.configs.shared_configs import TTSConfig

# FastAPI 앱 초기화
app = FastAPI(title="YourTTS 서버", description="YourTTS 모델을 사용한 TTS API 서버")

# 템플릿 및 정적 파일 설정
templates_dir = Path("templates")
static_dir = Path("static")

# 디렉토리가 없으면 생성
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

# 템플릿 및 정적 파일 설정
templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 테스트용 HTML 템플릿 생성
with open(templates_dir / "index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>YourTTS - 텍스트 음성 변환</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        textarea, select, input[type="text"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 16px;
        }
        textarea {
            height: 100px;
            resize: vertical;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .player {
            margin-top: 20px;
            width: 100%;
        }
        #loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>YourTTS - 텍스트 음성 변환</h1>
    
    <div class="form-group">
        <label for="text">텍스트:</label>
        <textarea id="text" required placeholder="변환할 텍스트를 입력하세요..."></textarea>
    </div>
    
    <div class="form-group">
        <label for="speaker">화자:</label>
        <select id="speaker">
            <option value="custom_voice">사용자 정의 음성</option>
        </select>
    </div>
    
    <div class="form-group">
        <label for="language">언어:</label>
        <select id="language">
            <option value="ko">한국어</option>
            <option value="en">영어</option>
            <option value="ja">일본어</option>
            <option value="zh-cn">중국어</option>
        </select>
    </div>
    
    <button onclick="generateSpeech()">음성 생성</button>
    
    <div id="loading">생성 중...</div>
    
    <div class="player" id="audioContainer" style="display:none;">
        <audio id="audioPlayer" controls style="width:100%"></audio>
    </div>
    
    <script>
        async function generateSpeech() {
            const text = document.getElementById('text').value;
            const speaker = document.getElementById('speaker').value;
            const language = document.getElementById('language').value;
            
            if (!text) {
                alert('텍스트를 입력해주세요.');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('audioContainer').style.display = 'none';
            
            try {
                const response = await fetch('/tts', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: text,
                        speaker_name: speaker,
                        language: language
                    })
                });
                
                if (!response.ok) {
                    throw new Error('음성 생성 실패');
                }
                
                const audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                
                const audioPlayer = document.getElementById('audioPlayer');
                audioPlayer.src = audioUrl;
                
                document.getElementById('audioContainer').style.display = 'block';
                audioPlayer.play();
            } catch (error) {
                console.error('에러:', error);
                alert('음성 생성 중 오류가 발생했습니다.');
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
    </script>
</body>
</html>
    """)

# 모델 및 설정
model = None
config = None
speakers = []

# API 요청 모델
class TTSRequest(BaseModel):
    text: str
    speaker_name: str = "custom_voice"
    language: str = "ko"
    speed: float = 1.0

# 모델 로드 함수
def load_model(model_path, config_path):
    global model, config, speakers
    
    # 설정 로드
    config = TTSConfig()
    config.load_json(config_path)
    
    # 모델 로드
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, model_path)
    
    # GPU가 있으면 GPU 사용
    if torch.cuda.is_available():
        model.cuda()
    
    model.eval()
    
    # 가능한 화자 목록 가져오기
    if hasattr(model, "speaker_manager") and model.speaker_manager is not None:
        speakers = model.speaker_manager.ids

    print(f"모델 로드 완료. 사용 가능한 화자: {speakers}")

# 루트 경로
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# TTS API 엔드포인트
@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    global model, config
    
    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
    
    try:
        # 화자가 유효한지 확인
        if request.speaker_name not in speakers and len(speakers) > 0:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 화자 이름입니다. 가능한 화자: {speakers}")
        
        # 텍스트 음성 변환 수행
        wav = model.inference(
            text=request.text,
            speaker_name=request.speaker_name,
            language_name=request.language,
            speed=request.speed
        )[0].cpu().numpy()
        
        # 사운드 파일로 변환
        wav = (wav * 32767).astype(np.int16)
        
        # 바이트 스트림으로 변환
        buffer = io.BytesIO()
        import scipy.io.wavfile as wavf
        wavf.write(buffer, rate=config.audio.sample_rate, data=wav)
        buffer.seek(0)
        
        # 스트리밍 응답 반환
        return StreamingResponse(buffer, media_type="audio/wav")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 처리 중 오류 발생: {str(e)}")

# 사용 가능한 화자 목록
@app.get("/speakers")
async def get_speakers():
    global speakers
    return {"speakers": speakers}

# 서버 시작 함수
def start_server(model_path, config_path, host, port):
    # 모델 로드
    load_model(model_path, config_path)
    
    # 서버 시작
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YourTTS API 서버')
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 모델 파일 경로')
    parser.add_argument('--config_path', type=str, required=True,
                        help='모델 설정 파일 경로')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='서버 호스트 (기본: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                        help='서버 포트 (기본: 8000)')
    
    args = parser.parse_args()
    
    # 서버 시작
    start_server(args.model_path, args.config_path, args.host, args.port) 