import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import time
import pandas as pd
import plotly.express as px
import numpy as np  

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="AI Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    .main { background-color: #0E1117; }
    h1, h2, h3 { color: #FAFAFA; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt") 

try:
    model = load_model()
except Exception as e:
    st.error(f"Không tìm thấy file model 'best.pt'. Lỗi: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌿 PLANT DIAGNOSIS")
    st.markdown("---")
    mode = st.radio("CHỌN CHẾ ĐỘ:", ["📸 Phân Tích Ảnh", "🎥 Phân Tích Video", "🔴 Live Camera"])
    st.markdown("---")
    conf_threshold = st.slider("Độ tin cậy (Confidence)", 0.0, 1.0, 0.25, 0.05)
    st.markdown("---")
    st.info("Hệ thống sử dụng YOLOv8\nOptimized for Streamlit")

# --- HÀM XỬ LÝ ---
def process_frame(frame, model, conf):
    # Resize nhẹ để tăng tốc (giống logic app cũ)
    height, width = frame.shape[:2]
    if height > 640:
        scale = 640 / height
        frame = cv2.resize(frame, (int(width * scale), 640))

    results = model.predict(frame, conf=conf, verbose=False)
    annotated_frame = results[0].plot()
    
    detections = []
    if results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            name = results[0].names[cls_id]
            detections.append({"Bệnh": name, "Độ tin cậy": conf_val})
            
    return annotated_frame, detections

# ==================================================
# MODE 1: ẢNH
# ==================================================
if mode == "📸 Phân Tích Ảnh":
    st.header("📂 Phân Tích Ảnh")
    uploaded_file = st.file_uploader("Chọn ảnh...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        image = Image.open(uploaded_file)
        img_array = np.array(image) # Giờ dòng này sẽ chạy ngon lành
        
        with st.spinner('Đang phân tích...'):
            result_img, detections = process_frame(img_array, model, conf_threshold)
        
        with col1:
            st.image(result_img, caption="Kết quả", use_container_width=True)
            
        with col2:
            st.subheader("Thống kê")
            if detections:
                df = pd.DataFrame(detections)
                st.dataframe(df, use_container_width=True)
                fig = px.bar(df, x='Độ tin cậy', y='Bệnh', orientation='h', color='Bệnh')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("Không phát hiện bệnh.")

# ==================================================
# MODE 2: VIDEO
# ==================================================
elif mode == "🎥 Phân Tích Video":
    st.header("Phân Tích Video")
    video_file = st.file_uploader("Chọn video...", type=['mp4', 'avi', 'mov'])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        col1, col2 = st.columns([3, 1])
        with col1: st_frame = st.empty()
        with col2: st_stat = st.empty()
            
        stop = st.button("Dừng Video")
        
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img, detections = process_frame(frame, model, conf_threshold)
            st_frame.image(result_img, channels="RGB", use_container_width=True)
            
            if detections:
                st_stat.dataframe(pd.DataFrame(detections).head(3), hide_index=True)

        cap.release()

# ==================================================
# MODE 3: CAMERA
# ==================================================
elif mode == "🔴 Live Camera":
    st.header("Live Camera")
    run = st.checkbox('BẮT ĐẦU', value=False)
    
    col1, col2 = st.columns([3, 1])
    with col1: st_frame = st.empty()
    with col2: st_stat = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        prev_time = 0
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Lỗi Camera")
                break
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img, detections = process_frame(frame, model, conf_threshold)
            
            cv2.putText(result_img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            st_frame.image(result_img, channels="RGB", use_container_width=True)
            
            if detections:
                st_stat.dataframe(pd.DataFrame(detections), hide_index=True)
                
        cap.release()