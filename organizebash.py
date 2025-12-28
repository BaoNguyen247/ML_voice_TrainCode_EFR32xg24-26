import os
import librosa
import numpy as np
import asyncio
import platform
import soundfile as sf
FPS = 60

async def main():
    # Đường dẫn thư mục chứa các thư mục con
    base_dir = "./adddata2/adddata"
    destdir = "./adddata2/output"
    target_duration_ms = 1000  # 1 giây = 1000ms
    sample_rate = 16000  # Tần số lấy mẫu mặc định, điều chỉnh nếu cần

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(destdir, exist_ok=True)

    # Duyệt qua tất cả các thư mục con
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            # Tạo thư mục con trong destdir
            dest_folder_path = os.path.join(destdir, folder)
            os.makedirs(dest_folder_path, exist_ok=True)

            # Đếm số thứ tự trong thư mục hiện tại
            file_count = 1
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(folder_path, file_name)

                    try:
                        # Đọc file WAV bằng librosa
                        audio, sr = librosa.load(file_path, sr=sample_rate)

                        # Tính độ dài hiện tại (tính bằng mili giây)
                        duration_ms = len(audio) * 1000 / sr

                        if duration_ms < target_duration_ms:
                            # Thêm silence nếu file ngắn hơn 1 giây
                            silence_samples = int((target_duration_ms - duration_ms) * sr / 1000)
                            silence_before = silence_samples // 2
                            silence_after = silence_samples - silence_before
                            audio = np.pad(audio, (silence_before, silence_after), mode='constant')
                        elif duration_ms > target_duration_ms:
                            # Cắt ngắn nếu file dài hơn 1 giây
                            target_samples = int(target_duration_ms * sr / 1000)
                            # Cắt từ giữa file để giữ phần trung tâm
                            start_sample = (len(audio) - target_samples) // 2
                            audio = audio[start_sample:start_sample + target_samples]

                        # Tạo tên file mới theo định dạng [tên_thư_mục]_[số_thứ_tự].wav
                        new_file_name = f"{folder}_{file_count}.wav"
                        # Ghi file vào thư mục con tương ứng trong destdir
                        output_path = os.path.join(dest_folder_path, new_file_name)
                        sf.write(output_path, audio, sr)
                        file_count += 1
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                        continue

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())