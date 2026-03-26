import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import os
import numpy as np
from datetime import datetime

# ================================
# STREAMLIT PAGE CONFIG & AESTHETICS
# ================================
st.set_page_config(page_title="Garbage Identify AI", layout="wide", page_icon="♻️")

# CSS Styling to give it a modern look
st.markdown("""
    <style>
    .main {background-color: #f4f6f9;}
    h1 {color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    h2, h3 {color: #34495e;}
    .stButton>button {background-color: #1abc9c; color: white; font-weight: bold; border-radius: 8px;}
    .stButton>button:hover {background-color: #16a085;}
    .reportview-container {background-color: #f4f6f9;}
    .stFileUploader>div>div>button {background-color: #3498db; color: white; border-radius: 8px;}
    .stFileUploader>div>div>button:hover {background-color: #2980b9;}
    .result-box {background-color: #ffffff; color: #2c3e50; padding: 20px; border-radius: 10px; border-left: 5px solid #1abc9c; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); font-family: 'Consolas', monospace; font-size: 16px;}
    </style>
""", unsafe_allow_html=True)

# ================================
# CONSTANTS & GLOBAL VARS
# ================================
DETECTION_DIR = "detections"
if not os.path.exists(DETECTION_DIR):
    os.makedirs(DETECTION_DIR)

WASTE_GROUPS = {
    "Rác thải nhựa ♻️ (0-3)": [0, 1, 2, 3],
    "Rác thải rắn 🧱 (4-8)": [4, 5, 6, 7, 8],
    "Nylon 🛍️ (9)": [9],
    "Rác thải y tế 💉 (10-13)": [10, 11, 12, 13]
}

# Load Model Path dynamically
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ai_model', 'ModelAI.pt'))

# ================================
# HELPER FUNCTIONS (CACHED)
# ================================
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        try:
            model = YOLO(path)
            return model, "✅ Model YOLOv8 đã tải thành công!"
        except Exception as e:
            return None, f"❌ Lỗi tải model: {e}"
    else:
        return None, "⚠️ Cảnh báo: Không tìm thấy file ModelAI.pt"

model, status_text = load_model(MODEL_PATH)

def letterbox_image(img, target_size=(780, 480), color=(0, 0, 0)):
    """Resize ảnh giữ nguyên tỷ lệ, phần thừa sẽ được bù đắp bằng viền đen (Letterbox) để luôn vừa khung tĩnh."""
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    return cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def delete_history_files(img_path, txt_path):
    """Xóa file ảnh và file txt tương ứng từ thư mục lịch sử."""
    try:
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(txt_path): os.remove(txt_path)
    except Exception as e:
        pass # Streamlit sẽ tự động bỏ qua nếu gặp lỗi truy xuất

def get_next_filename():
    """Tạo tên file theo thứ tự detect_001.jpg"""
    files = [f for f in os.listdir(DETECTION_DIR) if f.startswith("detect_") and f.endswith(".jpg")]
    if not files:
        count = 1
    else:
        counts = [int(f.split("_")[1].split(".")[0]) for f in files if "_" in f]
        count = max(counts) + 1 if counts else 1
    return os.path.join(DETECTION_DIR, f"detect_{count:03d}.jpg")

def format_results(results, model_instance):
    """Phân loại kết quả như Tkinter (Hiển thị nhãn, không đếm tổng số)"""
    all_class_ids = [int(box.cls[0]) for r in results for box in r.boxes]
    unique_class_ids = set(all_class_ids)

    grouped_results = {}
    for group_name, group_ids in WASTE_GROUPS.items():
        detected_labels = []
        for class_id in group_ids:
            if class_id in unique_class_ids:
                label = model_instance.names.get(class_id, f"Class {class_id}")
                if label not in detected_labels:
                    detected_labels.append(label)

        if detected_labels:
            grouped_results[group_name] = [f"• {label}" for label in detected_labels]

    if not all_class_ids or not grouped_results:
        return "Không phát hiện rác thải."

    display_text = ""
    for group_name, items in grouped_results.items():
        display_text += f"\n--- {group_name} ---\n"
        display_text += "\n".join(items) + "\n"

    return display_text.strip()

def save_detection(annotated_frame, formatted_text):
    """Lưu trữ ảnh và file txt"""
    output_path = get_next_filename()
    cv2.imwrite(output_path, annotated_frame)
    try:
        txt_path = os.path.splitext(output_path)[0] + ".txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        return output_path
    except Exception as e:
        st.error(f"Lỗi lưu file txt: {e}")
        return output_path

# ================================
# UI LAYOUT
# ================================
st.title("Garbage Identify Nhận diện & Phân loại Rác thải bằng AI")

if model:
    st.sidebar.success(status_text)
else:
    st.sidebar.error(status_text)
    st.stop()

# TABS
tab_detect, tab_history = st.tabs(["📸 Nhận Diện Rác Thải", "🕰️ Xem Lịch Sử"])

# ----------------
# TAB 1: DETECTION
# ----------------
with tab_detect:
    camera_mode = st.radio("Chọn phương thức nhận diện:", ("Live Camera", "Tải ảnh lên (Upload)"), horizontal=True)

    if camera_mode == "Live Camera":
        st.subheader("Trực Tiếp Video & Điều Khiển")
        
        # Đưa bảng chọn thiết lập Lên trên cùng để Không đẩy Nút Chụp xuống thấp
        col_c1, col_c2, col_c3 = st.columns([2, 2, 1])
        with col_c1:
            source_option = st.selectbox("Chọn Nguồn Camera", ["Webcam Laptop (0)", "Virtual Webcam (1)", "Link IP Điện thoại (HTTP)"])
        
        cap_src = None
        with col_c2:
            if source_option == "Webcam Laptop (0)": 
                cap_src = 0
            elif source_option == "Virtual Webcam (1)": 
                cap_src = 1
            else: 
                ip_link = st.text_input("Nhập Link IP (Ví dụ: http://.../video)", value="")
                if ip_link.strip():
                    cap_src = ip_link.strip()
                    
        with col_c3:
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            start_cam = st.checkbox("Bật Camera Stream")
            
        # Chia 2 Cột: Bên trái là Video -- Bên phải là nút Chụp hình và Kết quả
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Khung chứa video
            frame_placeholder = st.empty()
            
        with col_right:
            st.markdown("**Hành Động**")
            # st.button returns True in the rerun exactly once when clicked
            capture_btn = st.button("📸 Chụp Ảnh Hiện Tại", use_container_width=True)
            result_placeholder = st.empty()
            if not start_cam:
                result_placeholder.markdown(f"<div class='result-box'>Chưa khởi động Video.</div>", unsafe_allow_html=True)

            
            if capture_btn:
                st.session_state.trigger_capture = True
        
        # Vòng lặp Video Streaming
        if start_cam and cap_src is not None:
            # We use a session state to safely init VideoCapture
            cap = cv2.VideoCapture(cap_src)
            if not cap.isOpened():
                frame_placeholder.error("Không thể kết nối đến Camera ở nguồn này. Xin hãy tắt Bật Camera Stream, kiểm tra lại luồng và thử lại.")
            else:
                while cap.isOpened() and start_cam:
                    ret, frame = cap.read()
                    if not ret:
                        frame_placeholder.error("Bị ngắt kết nối Frame video.")
                        break
                    
                    # Cân chỉnh Hình ảnh Camera tự động ngay trong Code
                    if source_option == "Webcam Laptop (0)":
                        # Chỉ lật Mirror cho Cams trước
                        frame = cv2.flip(frame, 1)

                    elif source_option == "Virtual Webcam (1)":
                        # Dành cho Camera điện thoại Droidcam
                        # Xoay Video 90 độ để đứng đúng khung nhìn điện thoại cầm dọc
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                        # Không flip vì camera điện thoại phía sau không cần mirror
                    
                    # Cải tiến thuật toán hiển thị: KHÔNG resize cứng về 780x480 để giữ nguyên tỷ lệ thật của khung hình (Tránh bóp dẹp)
                    # YOLOv8 tự động resize bên trong khi dự đoán nên việc đưa nguyên khung gốc sẽ giữ được độ nét tự nhiên.
                    results = model(frame, conf=0.25, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    # Tạo Box hiển thị cố định có viền đen để không bị lật layout web
                    display_frame = letterbox_image(annotated_frame, target_size=(780, 480))
                    
                    # Convert màu BGR sang RGB cho Streamlit display
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Lấy nhãn
                    formatted_text = format_results(results, model)
                    
                    # Nếu có nút chụp
                    if st.session_state.get("trigger_capture"):
                        saved_path = save_detection(annotated_frame, formatted_text)
                        st.session_state.trigger_capture = False # reset flag
                        st.toast(f"Đã chụp ảnh và lưu thành {os.path.basename(saved_path)}!")
                    
                    # Render Frame lên Web
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Render Kết quả
                    html_result = formatted_text.replace('\\n', '<br>')
                    if "Không phát hiện" in formatted_text:
                        result_placeholder.markdown(f"<div class='result-box' style='border-left-color: #95a5a6;'>Không phát hiện vật thể.</div>", unsafe_allow_html=True)
                    else:
                        formatted_html_text = formatted_text.replace('\n', '<br>')
                        result_placeholder.markdown(f"<div class='result-box'>{formatted_html_text}</div>", unsafe_allow_html=True)
                        
            # Release memory if loop breaks
            if cap:
                cap.release()

    else:
        # CHẾ ĐỘ UPLOAD ẢNH
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Tải Ảnh Nhận Diện")
            uploaded_file = st.file_uploader("Chọn ảnh rác thải (JPG, PNG)", type=['jpg', 'jpeg', 'png', 'bmp'])
            
            if uploaded_file is not None:
                # Đọc byte từ Upload
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1) # BGR
                
                # Predict
                # Bỏ resize cố định (780x480) để giữ nguyên tỷ lệ Aspect Ratio hiển thị của các loại bức ảnh dọc/ngang ngẫu nhiên
                results = model(frame, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()
                
                formatted_text = format_results(results, model)
                
                # Hiển thị với khung cố định
                display_frame = letterbox_image(annotated_frame, target_size=(780, 480))
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB", use_container_width=True)
                
                with col_right:
                    st.subheader("Kết Quả Phân Loại")
                    if "Không phát hiện" in formatted_text:
                        st.markdown(f"<div class='result-box' style='border-left-color: #95a5a6;'>Không phát hiện vật thể.</div>", unsafe_allow_html=True)
                    else:
                        formatted_html_text = formatted_text.replace('\n', '<br>')
                        st.markdown(f"<div class='result-box'>{formatted_html_text}</div>", unsafe_allow_html=True)
                    
                    if st.button("💾 Lưu Dữ Liệu Phát Hiện Này"):
                        saved_path = save_detection(annotated_frame, formatted_text)
                        st.success(f"Đã lưu xử lý vào: {os.path.basename(saved_path)}")

# ----------------
# TAB 2: HISTORY
# ----------------
with tab_history:
    st.subheader("📌 Lịch sử Phát hiện Rác thải")
    st.write("Dưới đây là các ảnh bạn đã chụp hoặc lưu lại từ hệ thống.")
    
    # Load files từ detection_dir
    if not os.path.exists(DETECTION_DIR) or not os.listdir(DETECTION_DIR):
        st.info("Chưa có dữ liệu nào được lưu.")
    else:
        # Get only image files
        image_files = sorted(
            [f for f in os.listdir(DETECTION_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: os.path.getctime(os.path.join(DETECTION_DIR, x)),
            reverse=True # Mới nhất xếp trước
        )
        
        if not image_files:
             st.info("Chưa có hình ảnh nhận diện nào.")
        else:
            # Chia cột (Grid Layout) giống Tkinter Cols=4
            cols = st.columns(4)
            for idx, filename in enumerate(image_files):
                path = os.path.join(DETECTION_DIR, filename)
                txt_path = os.path.splitext(path)[0] + ".txt"
                
                creation_time = datetime.fromtimestamp(os.path.getctime(path)).strftime('%H:%M %d/%m/%Y')
                
                # Tìm cột để render dựa trên idx % 4
                col = cols[idx % 4]
                
                with col:
                    try:
                        img = Image.open(path)
                        # Render Image as button workaround (Streamlit limits image buttons natively, so we render image + expander for info)
                        st.image(img, use_container_width=True, caption=f"{filename}")
                        
                        with st.expander("Xem chi tiết"):
                            st.write(f"⏱️ {creation_time}")
                            
                            # Đọc text 
                            info_text = "Không phát hiện rác thải."
                            if os.path.exists(txt_path):
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    info_text = f.read()
                                    
                            st.markdown(f"**Kết quả:**\n\n{info_text}")
                            
                            # Hiển thị nút Tải và nút Xoá nằm chung hàng
                            col_btn1, col_btn2 = st.columns(2)
                            
                            with col_btn1:
                                # Download option
                                with open(path, "rb") as file:
                                    st.download_button(
                                        label="⬇️ Tải ảnh",
                                        data=file,
                                        file_name=filename,
                                        mime="image/jpeg",
                                        key=f"dl_{idx}",
                                        use_container_width=True
                                    )
                                    
                            with col_btn2:
                                # Delete option
                                st.button(
                                    "🗑️ Xóa", 
                                    key=f"del_{idx}", 
                                    on_click=delete_history_files, 
                                    args=(path, txt_path),
                                    use_container_width=True
                                )
                    except Exception as e:
                        st.error("Error loading img")
