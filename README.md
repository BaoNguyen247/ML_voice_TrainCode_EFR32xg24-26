# Findbot - Keyword Spotting Model Training

Dự án Findbot là một hệ thống huấn luyện mô hình **Keyword Spotting (KWS)** sử dụng công nghệ **Knowledge Distillation**. Mô hình được thiết kế để phát hiện các từ khóa ("on" và "off") từ tín hiệu âm thanh, có thể triển khai trên các thiết bị nhúng với tài nguyên hạn chế.

## 📋 Mục Lục
- [Tính Năng](#tính-năng)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Quá Trình Huấn Luyện](#quá-trình-huấn-luyện)
- [Các Tệp Output](#các-tệp-output)
- [Ghi Chú Quan Trọng](#ghi-chú-quan-trọng)

## 🎯 Tính Năng

- **Huấn luyện mô hình hai giai đoạn**: 
  - **Teacher Model**: Mô hình lớn, độ chính xác cao (48 filters)
  - **Student Model**: Mô hình nhỏ, tối ưu cho thiết bị nhúng (10-20 filters)
- **Knowledge Distillation**: Chuyển giao kiến thức từ mô hình giáo viên sang mô hình học sinh
- **Xuất mô hình TFLite**: Hỗ trợ triển khai trên các thiết bị nhúng (Edge devices)
- **Tập dữ liệu Speech Commands v2**: Dữ liệu âm thanh lớn, đa dạng
- **GPU hỗ trợ**: Tối ưu hóa với TensorFlow GPU

## 🔧 Yêu Cầu Hệ Thống

### Phần Cứng
- **GPU** (khuyến nghị): NVIDIA GPU với CUDA support
- **RAM**: Tối thiểu 8GB (16GB khuyến nghị)
- **Dung lượng ổ đĩa**: 50GB+ (cho dữ liệu huấn luyện)

### Phần Mềm
- **Python**: 3.12
- **MLTK**: 0.20
- **TensorFlow**: 2.x (có trong requirements.txt)

## 📦 Cài Đặt

### 1. Chuẩn Bị Môi Trường Python

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
venv\Scripts\activate

# Kích hoạt môi trường (Linux/Mac)
source venv/bin/activate
```

### 2. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 3. Chuẩn Bị Dữ Liệu

Tải file `ngonluadata.zip` và giải nén:
Link file: [https://drive.google.com/file/d/1qWuseka9wIfLXe-FJcniTyKoOhruPU3K/view?usp=drive_link]

**Trên Windows:**
- Giải nén file `ngonluadata.zip`
- Di chuyển thư mục `.mltk` vào `C:\` sao cho đường dẫn là: `C:\.mltk\`

**Trên Linux/Mac:**
- Giải nén file `ngonluadata.zip`
- Di chuyển thư mục `.mltk` vào thư mục gốc hoặc vị trí được chỉ định

> **Lưu ý**: Nếu lần đầu chạy code, hệ thống sẽ tự động tải dữ liệu về. Kiểm tra log để xác định vị trí `.mltk` phù hợp.

## 📁 Cấu Trúc Dự Án

```
findbot_colab_train/
├── findbot_colab_train.py              # File chính - định nghĩa mô hình
├── findbot_colab_train.teacher.h5      # Mô hình Teacher đã huấn luyện
├── requirements.txt                     # Danh sách dependencies
├── README.md                           # Tài liệu này
├── organizebash.py                     # Utility script
├── retype.py                           # Utility script
│
└── findbot_colab_train.mltk/           # Thư mục dự án MLTK
    ├── findbot_colab_train.tflite      # Mô hình TFLite (Student) - sản phẩm chính
    ├── findbot_colab_train.float32.tflite  # Mô hình TFLite (full precision)
    ├── findbot_colab_train.h5          # Mô hình Keras (Student)
    ├── findbot_colab_train.teacher.h5  # Mô hình Keras (Teacher)
    ├── findbot_colab_train.tflite.summary.txt
    │
    ├── dataset/                        # Dữ liệu huấn luyện
    │   └── .index/                     # Cache index (được gen tự động lần đầu)
    │
    └── train/                          # Kết quả huấn luyện
        └── log.txt
        └── training-history.json       # Lịch sử huấn luyện
```

### Chi Tiết Các Thư Mục MLTK

| Thư Mục | Mô Tả |
|---------|-------|
| `cli_logs/` | Chứa log từ quá trình huấn luyện, dùng để debug lỗi |
| `downloads/` | Tệp dữ liệu âm thanh được tải về - **KHÔNG XÓA** |
| `models/.../dataset/` | Thư mục chứa dữ liệu huấn luyện |
| `datasets/speech_commands/v2_cleaned/` | Tập dữ liệu chính (các folder sản phẩm, nhiễu, unknown với file audio.wav) |

## 🚀 Hướng Dẫn Sử Dụng

### Huấn Luyện Mô Hình Teacher (Giai Đoạn 1)

**Mô hình Teacher** là mô hình lớn, phức tạp, được sử dụng để tạo kiến thức cho mô hình Student.

#### Trên Windows (Command Prompt):

```cmd
# Thiết lập biến môi trường
set TRAIN_TEACHER=1

# Chạy huấn luyện
mltk train findbot_colab_train.py
```

#### Trên Windows (PowerShell):

```powershell
# Thiết lập biến môi trường
$env:TRAIN_TEACHER=1

# Chạy huấn luyện
mltk train findbot_colab_train.py
```

#### Trên Linux/Mac:

```bash
export TRAIN_TEACHER=1
mltk train findbot_colab_train.py
```

### Huấn Luyện Mô Hình Student (Giai Đoạn 2)

**Mô hình Student** là mô hình nhỏ, tối ưu được sinh ra từ kiến thức của Teacher Model.

#### Trên Windows (Command Prompt):

```cmd
set TRAIN_TEACHER=0
mltk train findbot_colab_train.py
```

#### Trên Windows (PowerShell):

```powershell
$env:TRAIN_TEACHER=0
mltk train findbot_colab_train.py
```

#### Trên Linux/Mac:

```bash
export TRAIN_TEACHER=0
mltk train findbot_colab_train.py
```

## 📊 Quá Trình Huấn Luyện

### Cấu Hình Mô Hình

| Tham Số | Giá Trị | Mô Tả |
|---------|--------|-------|
| **Epochs** | 75 | Số lần lặp qua toàn bộ tập dữ liệu |
| **Batch Size** | 32 | Số mẫu xử lý cùng lúc |
| **Version** | 2 | Phiên bản mô hình |
| **Loss Function** | Categorical Crossentropy | Hàm mất mát cho phân loại đa lớp |
| **Optimizer** | Adam (learning_rate=0.001) | Thuật toán tối ưu |

### Kiến Trúc Mô Hình

**Teacher Model** (48 filters):
- 5 lớp Convolutional với BatchNormalization
- MaxPooling để giảm kích thước
- Dropout (0.5) để chống overfitting
- Output: Softmax classification layer

**Student Model** (10-20 filters):
- Kiến trúc tương tự nhưng nhỏ gọn hơn
- Dropout (0.3)
- Tối ưu cho triển khai trên thiết bị nhúng

## 📤 Các Tệp Output

Sau khi huấn luyện, các tệp sau sẽ được sinh ra:

| Tệp | Mô Tả |
|-----|-------|
| `findbot_colab_train.tflite` | **Chính** - Mô hình TFLite nén (dùng cho triển khai) |
| `findbot_colab_train.float32.tflite` | Mô hình TFLite full precision |
| `findbot_colab_train.h5` | Mô hình Keras (định dạng nhị phân) |
| `findbot_colab_train.tflite.summary.txt` | Thông tin chi tiết về mô hình TFLite |
| `training-history.json` | Lịch sử accuracy/loss qua các epoch |
| `log.txt` | Log huấn luyện chi tiết |

## ⚠️ Ghi Chú Quan Trọng

### Cache Index Dataset

Khi chạy code lần đầu, một thư mục `.index` được tạo tự động trong `dataset/`. Thư mục này chứa **thông tin của tất cả dữ liệu huấn luyện**.

- **Lần chạy tiếp theo**: Mô hình sẽ sử dụng `.index` này làm input, không tải lại dữ liệu
- **Nếu có sai sót lần đầu**: Xóa thư mục `.index` để regenerate
- **Lưu ý**: Không xóa các file dữ liệu trong `downloads/` vì sẽ cần tải lại (mất thời gian và có thể gặp lỗi link)

### Lời Khuyên Khi Gặp Vấn Đề

1. **Lỗi tải dữ liệu**: Kiểm tra `cli_logs/` để xem log chi tiết
2. **Vấn đề bộ nhớ**: Giảm `batch_size` trong [findbot_colab_train.py](findbot_colab_train.py#L58)
3. **Lỗi đường dẫn `.mltk`**: Chạy code một lần để nó tự tạo `.mltk`, sau đó kiểm tra log để định vị chính xác

### Biến Môi Trường

Biến `TRAIN_TEACHER` chỉ có hiệu lực trong **phiên terminal hiện tại**. Nếu đóng terminal, cần thiết lập lại biến.

## 📝 Dependencies Chính

```
tensorflow          # Framework deep learning
mltk               # Machine Learning Toolkit
keras              # Neural network API
numpy              # Tính toán số học
audiomentations    # Tăng cường dữ liệu âm thanh
google-cloud-storage  # Tải dữ liệu từ cloud
```

Chi tiết đầy đủ: Xem [requirements.txt](requirements.txt)

## 📌 Lệnh Nhanh

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Huấn luyện Teacher (Windows CMD)
set TRAIN_TEACHER=1 & mltk train findbot_colab_train.py

# Huấn luyện Student (Windows CMD)
set TRAIN_TEACHER=0 & mltk train findbot_colab_train.py

# Huấn luyện Teacher (Linux/Mac)
TRAIN_TEACHER=1 mltk train findbot_colab_train.py

# Huấn luyện Student (Linux/Mac)
TRAIN_TEACHER=0 mltk train findbot_colab_train.py
```

## 📧 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra file log trong `cli_logs/`
2. Xem phần [Ghi Chú Quan Trọng](#ghi-chú-quan-trọng) ở trên
3. Đảm bảo Python version 3.12 và MLTK 0.20 được cài đặt đúng

---

**Cập nhật lần cuối**: Tháng 12, 2025  
**Phiên bản**: v2
