import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# Đường dẫn chứa ảnh của các thành viên
dataset_path = r"D:\insight\nhan_dien_khuon_mat\dataset"

# Khởi tạo InsightFace để trích xuất khuôn mặt
face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                        allowed_modules=['detection', 'recognition'])  
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Dictionary lưu embeddings và thông tin sinh viên
face_db = {}

# Duyệt qua từng ảnh trong dataset
for img_name in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_name)

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        print(f"Lỗi đọc ảnh {img_name}")
        continue

    # Phát hiện khuôn mặt
    faces = face_app.get(img)
    if len(faces) == 0:
        print(f"Không tìm thấy khuôn mặt trong ảnh {img_name}")
        continue

    # Lấy đặc trưng khuôn mặt (embedding)
    face_embedding = faces[0].embedding

    # Tách thông tin từ tên file (MãSV, Họ tên, Lớp)
    try:
        student_id, name, class_name = img_name[:-4].split("_")
    except ValueError:
        print(f"Lỗi: Tên file '{img_name}' không đúng định dạng!")
        continue

    # Nếu student_id đã có, thêm embedding mới vào danh sách
    if student_id not in face_db:
        face_db[student_id] = {
            "name": name,
            "class": class_name,
            "embeddings": []  # Danh sách lưu nhiều embeddings
        }

    # Thêm embedding mới vào danh sách
    face_db[student_id]["embeddings"].append(face_embedding)

# Lưu embeddings vào file
with open("face_db.pkl", "wb") as f:
    pickle.dump(face_db, f)

print("Dataset đã được xử lý và lưu vào 'face_db.pkl'")
