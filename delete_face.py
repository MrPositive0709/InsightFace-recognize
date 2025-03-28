import numpy as np
import os

DB_PATH = "face_database.npz"

def delete_face(name_to_delete):
    if not os.path.exists(DB_PATH):
        print("Không tìm thấy database.")
        return
    
    # Load dữ liệu từ file npz
    data = np.load(DB_PATH, allow_pickle=True)
    names = data["names"].tolist()
    embeddings = data["embeddings"].tolist()
    
    # Kiểm tra xem có tên trong database không
    if name_to_delete not in names:
        print(f"Không tìm thấy {name_to_delete} trong database.")
        return
    
    # Xóa dữ liệu tương ứng
    index = names.index(name_to_delete)
    del names[index]
    del embeddings[index]

    # Lưu lại database mới
    np.savez(DB_PATH, names=np.array(names), embeddings=np.array(embeddings))
    print(f"Đã xóa {name_to_delete} khỏi database.")

if __name__ == "__main__":
    delete_face("face_0")  # Thay "face_0" bằng tên bạn muốn xóa
