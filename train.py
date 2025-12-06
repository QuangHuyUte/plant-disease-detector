from ultralytics import YOLO

# 1. Load model
# Dùng bản 'nano' (n) cho nhẹ máy. Nếu máy mạnh (có GPU xịn) thì đổi thành 'yolov8s.pt'
model = YOLO('yolov8n.pt') 

if __name__ == '__main__':
    # 2. Train model
    # epochs=30: Học 30 lần (bạn có thể tăng lên 50-100 nếu muốn xịn hơn)
    # imgsz=640: Kích thước ảnh chuẩn
    # data: Trỏ vào file yaml bạn vừa sửa
    results = model.train(
        data='datasets/data.yaml', 
        epochs=30, 
        imgsz=640,
        project='runs/detect',
        name='plantdoc_model'
    )
    
    print("✅ Train xong! Kết quả lưu ở: runs/detect/plantdoc_model/weights/best.pt")