import os
import csv
import argparse
import librosa
import soundfile as sf
from pydub import AudioSegment
from pathlib import Path

def convert_to_wav(audio_path, output_path, sample_rate=22050):
    """다양한 오디오 형식을 WAV로 변환"""
    try:
        if not audio_path.endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            audio.export(output_path, format='wav')
        else:
            # 이미 WAV 파일인 경우 리샘플링만 수행
            y, sr = librosa.load(audio_path, sr=sample_rate)
            sf.write(output_path, y, sample_rate)
        return True
    except Exception as e:
        print(f"오류: {audio_path} 변환 중 문제 발생 - {e}")
        return False

def prepare_metadata(wavs_dir, output_csv):
    """메타데이터 CSV 파일 생성"""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'text', 'speaker_name', 'language'])
        
        wav_files = [f for f in os.listdir(wavs_dir) if f.endswith('.wav')]
        speaker_name = "custom_voice"  # 기본 화자 이름
        language = "ko"  # 한국어
        
        for wav_file in wav_files:
            file_path = os.path.join('wavs', wav_file)
            # 텍스트는 추후 수동으로 업데이트해야 할 수 있습니다
            placeholder_text = "이 오디오 파일의 텍스트 내용입니다."
            writer.writerow([file_path, placeholder_text, speaker_name, language])
    
    print(f"{len(wav_files)}개 오디오 파일에 대한 메타데이터가 {output_csv}에 저장되었습니다.")
    print("참고: 정확한 텍스트 전사가 필요한 경우 CSV 파일을 수동으로 업데이트하세요.")

def main():
    parser = argparse.ArgumentParser(description='YourTTS를 위한 데이터셋 준비')
    parser.add_argument('--audio_dir', type=str, default='source_audio',
                        help='원본 오디오 파일이 있는 디렉토리')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='처리된 WAV 파일과 메타데이터를 저장할 디렉토리')
    args = parser.parse_args()
    
    source_dir = Path(args.audio_dir)
    output_wavs_dir = Path(args.output_dir) / 'wavs'
    metadata_path = Path(args.output_dir) / 'metadata.csv'
    
    # 출력 디렉토리 생성
    os.makedirs(output_wavs_dir, exist_ok=True)
    
    # 오디오 파일이 없는 경우 안내
    if not source_dir.exists():
        print(f"'{source_dir}' 디렉토리를 찾을 수 없습니다.")
        print(f"'{output_wavs_dir}' 디렉토리에 WAV 파일을 직접 넣고 메타데이터만 생성합니다.")
        prepare_metadata(output_wavs_dir, metadata_path)
        return
    
    # 오디오 파일 변환
    processed_files = 0
    for audio_file in source_dir.glob('*'):
        if audio_file.is_file():
            output_path = output_wavs_dir / f"{audio_file.stem}.wav"
            if convert_to_wav(str(audio_file), str(output_path)):
                processed_files += 1
    
    print(f"{processed_files}개 파일이 WAV 형식으로 변환되었습니다.")
    
    # 메타데이터 생성
    prepare_metadata(output_wavs_dir, metadata_path)

if __name__ == "__main__":
    main()
    print("\n데이터셋 준비가 완료되었습니다!")
    print("다음 단계: train.py를 실행하여 YourTTS 모델을 학습하세요.") 