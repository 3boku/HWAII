# YourTTS 모델 학습 및 사용 가이드

이 가이드는 YourTTS를 사용하여 사용자 음성을 학습하고 FastAPI로 TTS 웹 서버를 실행하는 방법을 안내합니다.

## 1. 환경 설정

### 필수 요구사항

-   Python 3.7 이상
-   NVIDIA GPU (RTX 3070 GDDR6 8GB 최적화됨)
-   CUDA 11.0 이상

### 가상환경 설정

```bash
# 1. 가상환경 활성화 (Git Bash 방식)
source venv/Scripts/activate

# 또는 Windows CMD에서:
# venv\Scripts\activate.bat

# 2. 필요한 파일들이 있는 voice_model 디렉토리로 이동
cd ../voice_model

# 3. requirements.txt 파일이 있는지 확인
ls requirements.txt

# 4. 파일이 있다면 설치 진행
pip install -r requirements.txt
```

## 2. 음성 데이터 준비

### 데이터 구조

1. `source_audio` 폴더를 생성하고 학습하려는 음성 파일을 넣습니다.

    - 다양한 오디오 포맷을 지원합니다 (mp3, wav, m4a 등)
    - 각 오디오 파일은 5~10초 길이가 이상적입니다
    - 총 10분 이상의 음성 데이터가 권장됩니다 (파일 20개 이상)

2. 데이터 준비 스크립트 실행:

```bash
python prepare_dataset.py --audio_dir source_audio --output_dir data
```

3. 생성된 메타데이터 파일 (`data/metadata.csv`)을 확인하고 필요시 수정:
    - `file_path`: 오디오 파일 경로
    - `text`: 오디오 파일의 텍스트 내용
    - `speaker_name`: 화자 이름 (기본값: "custom_voice")
    - `language`: 언어 코드 (기본값: "ko")

## 3. 모델 학습

### 새 모델 학습 (처음부터)

```bash
python train.py --data_dir data --output_path output --run_name yourtts-korean
```

### 사전 학습된 모델에서 파인튜닝 (권장)

```bash
python train.py --data_dir data --output_path output --run_name yourtts-korean-finetuned --fine_tune
```

### 학습 매개변수 조정 (RTX 3070 8GB 최적화)

-   학습 스크립트에서 이미 RTX 3070에 최적화된 배치 사이즈(12)로 설정되어 있습니다.
-   메모리 부족 오류가 발생하면 배치 사이즈를 8로 줄여보세요.

### 학습 진행 확인

-   학습 로그는 터미널에 출력됩니다.
-   텐서보드를 사용하여 학습 진행 상황을 시각적으로 확인할 수 있습니다:

```bash
tensorboard --logdir=output
```

## 4. TTS 웹 서버 실행

### 서버 시작

```bash
python server.py --model_path output/best_model.pth --config_path output/config.json
```

### 서버 사용

1. 웹 브라우저에서 `http://localhost:8000`에 접속
2. 텍스트 입력, 화자 및 언어 선택
3. "음성 생성" 버튼을 클릭하여 TTS 생성

### API 사용 (프로그래밍 방식)

-   POST 요청을 `/tts` 엔드포인트로 보냅니다:

```json
{
    "text": "안녕하세요, 반갑습니다.",
    "speaker_name": "custom_voice",
    "language": "ko",
    "speed": 1.0
}
```

## 5. 문제 해결

### 일반적인 문제

1. **메모리 부족 오류**: 배치 사이즈를 줄이세요 (train.py에서 `batch_size` 값 수정)
2. **학습 속도가 느림**: 학습 데이터 양을 줄이거나 에폭 수를 줄이세요
3. **발음 품질 저하**: 더 많은 고품질 음성 데이터로 학습하세요

### 모델 품질 향상 팁

1. 조용한 환경에서 고품질 마이크로 녹음하세요
2. 일관된 톤과 속도로 말하세요
3. 다양한 문장과 표현이 포함된 데이터를 사용하세요

## 6. 참고 자료

-   [YourTTS 논문](https://arxiv.org/abs/2109.11737)
-   [Coqui TTS 문서](https://tts.readthedocs.io/)
-   [FastAPI 문서](https://fastapi.tiangolo.com/)
