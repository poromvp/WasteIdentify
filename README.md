# ♻️ Garbage Identify AI - Nhận diện & Phân loại Rác thải

Ứng dụng web thông minh sử dụng mô hình **YOLOv8** để nhận diện và phân loại các loại rác thải trong thời gian thực qua Camera hoặc ảnh tải lên. Project được phát triển bằng ngôn ngữ **Python** và thư viện **Streamlit**.

## ✨ Tính năng nổi bật
* [cite_start]**Nhận diện đa phương thức:** Hỗ trợ Live Camera (Webcam/IP Camera) và tải ảnh trực tiếp từ máy tính.
* **Phân loại thông minh:** Tự động nhóm các loại rác thành 4 nhóm chính:
    * ♻️ Rác thải nhựa
    * 🧱 Rác thải rắn
    * 🛍️ Nylon
    * [cite_start]💉 Rác thải y tế.
* [cite_start]**Lưu trữ lịch sử:** Cho phép chụp ảnh màn hình và lưu kết quả nhận diện kèm thời gian cụ thể để xem lại sau.
* [cite_start]**Giao diện hiện đại:** Tối ưu hóa trải nghiệm người dùng với CSS tùy chỉnh, bố cục trực quan.

## 🛠 Công nghệ sử dụng
* **AI Model:** YOLOv8 (Ultralytics).
* **Web Framework:** Streamlit.
* **Computer Vision:** OpenCV.
* [cite_start]**Language:** Python 3.12+.

## 🚀 Hướng dẫn cài đặt

### 1. Clone repository
```bash
git clone [https://github.com/poromvp/garbage-identify-ai.git](https://github.com/poromvp/garbage-identify-ai.git)
cd garbage-identify-ai

### 2. Cài đặt môi trường (Khuyên dùng venv)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

### 3. Cài đặt thư viện
pip install -r requirements.txt

Cách khởi chạy
streamlit run app.py