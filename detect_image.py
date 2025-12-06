import cv2
from ultralytics import YOLO
import os

# --- CẤU HÌNH ---
# 1. Đường dẫn đến file model vừa train xong
model_path = r'runs\detect\plantdoc_model2\weights\best.pt'

# 2. Tên file ảnh bạn muốn test (để cùng thư mục với file code này)
image_name = 'test.jpg'


# --- XỬ LÝ ---
# Kiểm tra xem file ảnh có tồn tại không
if not os.path.exists(image_name):
    print(f"❌ Lỗi: Không tìm thấy file ảnh '{image_name}'")
    print("👉 Hãy copy một tấm ảnh vào thư mục này và đổi tên thành 'test.jpg' nhé.")
    exit()

print("⏳ Đang load model và xử lý ảnh...")

# Load model
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    exit()

# Đọc ảnh vào bộ nhớ
img = cv2.imread(image_name)

# --- DỰ ĐOÁN (DETECT) ---
# conf=0.25: Chỉ hiện những cái khung nào máy tự tin >= 25%
# save=True: Tự động lưu ảnh kết quả vào thư mục runs/detect/predict
results = model.predict(img, conf=0.05, save=True)

# --- HIỂN THỊ KẾT QUẢ LÊN MÀN HÌNH ---
# Lấy ảnh đã được vẽ khung sẵn bởi thư viện YOLO
result_img = results[0].plot()

# Hiển thị cửa sổ
cv2.imshow("Ket qua Phat hien Benh", result_img)

print("✅ Đã xong! Hãy nhìn cửa sổ ảnh hiện lên.")
print("ℹ️ Ảnh kết quả cũng đã được lưu trong thư mục 'runs/detect/predict'")
print("👉 Nhấn một phím bất kỳ trên bàn phím để tắt cửa sổ.")

# Giữ màn hình đứng yên cho đến khi nhấn phím bất kỳ
cv2.waitKey(0)
cv2.destroyAllWindows()