import cv2
import numpy as np
from insightface.app import FaceAnalysis
import argparse
import os



# Thư mục lưu database
DB_PATH = "face_database.npz"

def detect_faces(image_path,save_dir, model_name ="buffalo_l"):
    

    # Khởi tạo ứng dụng FaceAnalysis với mô hình SCRFD
    app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể mở ảnh {image_path}")
        return
    
    # Chạy mô hình nhận diện khuôn mặt
    faces = app.get(image)

    embeddings = []
    names =[]

     # Xử lý từng khuôn mặt
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_img = image[y1:y2, x1:x2]  # Cắt khuôn mặt

        # Lưu khuôn mặt
        os.makedirs(save_dir, exist_ok=True)
        face_path = os.path.join(save_dir, f"face_{i}.jpg")
        cv2.imwrite(face_path, face_img)

        # Lưu bounding box trên ảnh gốc
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Trích xuất embedding
        embedding = face.embedding
        embeddings.append(embedding)
        names.append(f"Tâm")  # Đặt tên cho khuôn mặt
    
    # Lưu database khuôn mặt
    save_database(names, embeddings)


    # Vẽ bounding box lên ảnh
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save output
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, image)



def save_database(names, embeddings):
    """Lưu database người quen vào file .npz"""
    if os.path.exists(DB_PATH):
        data = np.load(DB_PATH, allow_pickle=True)
        existing_names = data["names"].tolist()
        existing_embeddings = data["embeddings"].tolist()
    else:
        existing_names = []
        existing_embeddings = []

    # Thêm dữ liệu mới
    existing_names.extend(names)
    existing_embeddings.extend(embeddings)

    # Lưu lại vào file npz
    np.savez(DB_PATH, names=np.array(existing_names), embeddings=np.array(existing_embeddings))
    print(f"Database đã được cập nhật với {len(names)} khuôn mặt.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="buffalo_l")
    parser.add_argument('--image_path', type=str, default="image1.jpg")
    parser.add_argument('--save_dir', type=str, default="output_database")
    
    args = parser.parse_args()

    detect_faces(args.image_path,"output_database","buffalo_l")