# YourTTS 음성 모델 학습 및 FastAPI 서버

이 프로젝트는 YourTTS를 사용하여 사용자 음성을 학습하고, FastAPI를 통해 TTS 웹 서버를 제공합니다.

## 설치 방법

```bash
# 가상환경 생성 및 활성화 (Windows)
python -m venv venv
venv\Scripts\activate

# 필요 패키지 설치
pip install -r requirements.txt
```

## 음성 데이터 준비

1. `data/wavs` 폴더에 학습시킬 음성 파일(WAV 형식)을 저장합니다.
2. `prepare_dataset.py` 스크립트를 실행하여 데이터셋을 준비합니다.

## 모델 학습

```bash
python train.py
```

## TTS 웹 서버 실행

```bash
python server.py
```

## 디렉토리 구조

```
voice_model/
├── data/                # 데이터 디렉토리
│   ├── wavs/            # 음성 파일 저장
│   └── metadata.csv     # 메타데이터 파일
├── prepare_dataset.py   # 데이터셋 준비 스크립트
├── train.py             # 모델 학습 스크립트
├── server.py            # FastAPI 서버
├── requirements.txt     # 필요 패키지 목록
└── README.md            # 현재 파일
```

## 참고사항

-   NVIDIA RTX 3070 GDDR6 8GB에 최적화된 배치 사이즈가 적용되어 있습니다.
-   YourTTS 학습에는 많은 시간이 소요될 수 있습니다.
