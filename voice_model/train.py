import os
import argparse
from pathlib import Path
from TTS.tts.configs.yourtts_config import YourTTSConfig
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import FineTuningManager
from TTS.utils.audio import AudioProcessor
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.trainer import Trainer, TrainingArgs

def prepare_yourtts_training(args):
    """YourTTS 모델 학습을 위한 설정"""
    
    # RTX 3070 8GB에 최적화된 배치 사이즈
    batch_size = 12  # GDDR6 8GB에 최적화
    
    # VITS 오디오 설정
    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        mel_fmin=0,
        mel_fmax=8000,
        num_mels=80
    )
    
    # 멀티스피커 설정
    model_args = VitsArgs(
        d_vector_dim=256,
        use_d_vector_file=True,
        d_vector_file=args.d_vector_file if args.d_vector_file else None,
        speaker_encoder_model_path=args.speaker_encoder if args.speaker_encoder else None,
        use_speaker_encoder_as_loss=False,
        speaker_encoder_config_path=None,
    )
    
    # 데이터셋 설정
    dataset_config = BaseDatasetConfig(
        formatter="nino",
        meta_file_train=str(Path(args.data_dir) / "metadata.csv"),
        path=args.data_dir,
        language="ko"
    )
    
    # YourTTS 설정
    config = YourTTSConfig(
        audio=audio_config,
        model_args=model_args,
        run_name=args.run_name,
        batch_size=batch_size,
        eval_batch_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        precompute_num_workers=4,
        compute_input_seq_cache=True,
        run_eval=True,
        test_delay_epochs=25,
        epochs=1000,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="ko",
        phoneme_cache_path=os.path.join(args.output_path, "phoneme_cache"),
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=args.output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        max_text_len=500,
        lr_scheduler="NoamLR",
        lr=0.0004,
        optimizer="AdamW",
        optimizer_params={
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "weight_decay": 0.01
        },
        lr_scheduler_params={
            "warmup_steps": 4000
        },
        scheduler_after_epoch=False,
        tensorboard=True,
        steps_to_start_discriminator=5000,
        target_loss="loss_1",
        use_language_embedding=True,
        use_speaker_embedding=True,
        min_text_len=1,
        min_audio_len=1,
        max_audio_len=160000,
        use_attn_loss=True,
    )
    
    return config

def train_yourtts(args):
    """YourTTS 모델 학습 시작"""
    # 디렉토리 생성
    os.makedirs(args.output_path, exist_ok=True)
    
    # YourTTS 설정 준비
    config = prepare_yourtts_training(args)
    
    # 데이터셋 로드
    train_samples, eval_samples = load_tts_samples(
        config.datasets[0].meta_file_train,
        eval_split_max_size=0.1,
        eval_split_size=0.1,
    )
    
    # 모델 학습
    if args.fine_tune and args.checkpoint:
        # 파인튜닝 설정
        manager = FineTuningManager(
            target_model_path=args.checkpoint,
            target_config_path=args.config,
            output_path=args.output_path,
        )
        
        # 파인튜닝 설정으로 업데이트
        new_config, new_model_path = manager.setup_fine_tuning(config)
        
        # 트레이너 초기화 및 학습 시작 (파인튜닝)
        trainer = Trainer(
            TrainingArgs(continue_path=new_model_path),
            config=new_config,
            output_path=args.output_path,
        )
    else:
        # 트레이너 초기화 및 학습 시작 (처음부터)
        trainer = Trainer(
            TrainingArgs(),
            config=config,
            output_path=args.output_path,
        )
    
    trainer.fit()
    
    print("모델 학습 완료!")
    print(f"모델이 {args.output_path}에 저장되었습니다.")

def download_pretrained_model(output_path):
    """사전 학습된 YourTTS 모델 다운로드"""
    from TTS.utils.manage import ModelManager
    
    # 모델 매니저 초기화
    model_manager = ModelManager()
    
    # YourTTS 모델 다운로드
    model_path, config_path, model_item = model_manager.download_model("tts_models/multilingual/multi-dataset/your_tts")
    
    return model_path, config_path

def main():
    parser = argparse.ArgumentParser(description='YourTTS 모델 학습')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='데이터셋 디렉토리')
    parser.add_argument('--output_path', type=str, default='output',
                        help='모델 출력 디렉토리')
    parser.add_argument('--run_name', type=str, default='yourtts-korean',
                        help='학습 실행 이름')
    parser.add_argument('--fine_tune', action='store_true',
                        help='사전 학습된 모델에서 파인튜닝 여부')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='파인튜닝을 위한 체크포인트 경로')
    parser.add_argument('--config', type=str, default=None,
                        help='파인튜닝을 위한 설정 파일 경로')
    parser.add_argument('--speaker_encoder', type=str, default=None,
                        help='스피커 인코더 모델 경로')
    parser.add_argument('--d_vector_file', type=str, default=None,
                        help='사전 계산된 d-vector 파일 경로')
    args = parser.parse_args()
    
    # 사전 학습된 모델로 파인튜닝하는 경우
    if args.fine_tune and not args.checkpoint:
        print("사전 학습된 YourTTS 모델 다운로드 중...")
        model_path, config_path = download_pretrained_model(args.output_path)
        args.checkpoint = model_path
        args.config = config_path
        print(f"다운로드된 모델: {model_path}")
        print(f"다운로드된 설정: {config_path}")
    
    # YourTTS 모델 학습 시작
    train_yourtts(args)
    
    print("\n학습이 완료되었습니다!")
    print("다음 단계: server.py를 실행하여 TTS 웹 서버를 시작하세요.")

if __name__ == "__main__":
    main() 