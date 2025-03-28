import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
from scipy.spatial.distance import cosine
import argparse

# Đường dẫn đến database
DB_PATH = "face_database.npz"

def load_database():
    """Tải dữ liệu khuôn mặt từ file npz"""
    if not os.path.exists(DB_PATH):
        print("❌ Database chưa tồn tại!")
        return [], []

    data = np.load(DB_PATH, allow_pickle=True)
    return data["names"].tolist(), data["embeddings"]

def recognize_face(image_path, model_name="buffalo_l", threshold=0.6):
    """Nhận diện khuôn mặt và kiểm tra xem có trong database không"""
    # Khởi tạo model
    app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load database
    names, embeddings = load_database()

    if not names:
        print("⚠️ Database trống. Hãy thêm khuôn mặt trước!")
        return

    # Đọc ảnh đầu vào
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Không thể mở ảnh {image_path}")
        return

    # Phát hiện khuôn mặt và trích xuất embedding
    faces = app.get(image)
    
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.embedding  # Trích xuất đặc trưng khuôn mặt

        # So sánh với database
        min_dist = float("inf")
        best_match = "Người lạ"

        for db_name, db_embedding in zip(names, embeddings):
            dist = cosine(db_embedding, embedding)  # Tính khoảng cách cosine
            if dist < min_dist:
                min_dist = dist
                best_match = db_name if dist < threshold else "Người lạ"

        # Hiển thị kết quả
        label = f"{best_match} ({min_dist:.2f})"
        color = (0, 255, 0) if best_match != "Người lạ" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save ảnh vào output_recognize
    os.makedirs("output_recognize", exist_ok=True)
    out_path = os.path.join("output_recognize", os.path.basename(image_path))
    cv2.imwrite(out_path, image)
    print(f"Kết quả đã được lưu vào {out_path}")
    



# Chạy kiểm tra với ảnh mới
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="Tam_front.jpg")  
    args = parser.parse_args()

    recognize_face(args.image_path,"buffalo_l",0.6)