import os
from pydub import AudioSegment


def convert_wav_automate(input_base_dir, output_base_dir, sample_rate=16000, bit_depth=16, channels=1):
    """
    Tự động chuyển đổi tất cả các tệp WAV trong các thư mục con của input_base_dir
    sang định dạng PCM 16-bit, mono, 16kHz và lưu vào output_base_dir.

    Args:
        input_base_dir (str): Thư mục gốc chứa các thư mục label với tệp WAV.
        output_base_dir (str): Thư mục gốc lưu các tệp WAV đã chuyển đổi.
        sample_rate (int): Tần số lấy mẫu (Hz), mặc định 16000.
        bit_depth (int): Độ sâu bit, mặc định 16 (cho PCM 16-bit).
        channels (int): Số kênh, mặc định 1 (mono).
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Duyệt qua tất cả các thư mục con trong input_base_dir
    for label_folder in os.listdir(input_base_dir):
        input_label_dir = os.path.join(input_base_dir, label_folder)
        if os.path.isdir(input_label_dir):
            # Tạo thư mục tương ứng trong output_base_dir
            output_label_dir = os.path.join(output_base_dir, label_folder)
            if not os.path.exists(output_label_dir):
                os.makedirs(output_label_dir)

            # Duyệt qua tất cả tệp WAV trong thư mục label
            for filename in os.listdir(input_label_dir):
                if filename.endswith(".wav"):
                    input_path = os.path.join(input_label_dir, filename)
                    output_path = os.path.join(output_label_dir, filename)

                    try:
                        # Đọc tệp WAV
                        audio = AudioSegment.from_wav(input_path)

                        # Chuyển đổi tần số lấy mẫu, số kênh, và độ sâu bit
                        audio = audio.set_frame_rate(sample_rate)
                        audio = audio.set_channels(channels)
                        audio = audio.set_sample_width(bit_depth // 8)  # Độ sâu bit tính bằng byte (16 bit = 2 byte)

                        # Lưu tệp WAV đầu ra
                        audio.export(output_path, format="wav")
                    except Exception as e:
                        print(f"Lỗi khi xử lý {label_folder}/{filename}: {e}")


if __name__ == "__main__":
    # Đường dẫn thư mục gốc chứa các thư mục label
    input_base_directory = "./data/legacy_16khz_data"
    # Đường dẫn thư mục gốc lưu kết quả chuyển đổi
    output_base_directory = "./data/legacy_16k_252kpbs_data"

    # Chạy chuyển đổi
    convert_wav_automate(input_base_directory, output_base_directory)