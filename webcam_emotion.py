import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque


MODEL_PATH = "E:\\HK1_Nam3\\TGMT\\emotion_model5.h5"
model = load_model(MODEL_PATH)

# Lấy input shape của model
input_shape = model.input_shape
img_height = input_shape[1]
img_width = input_shape[2]
channels = input_shape[3] if len(input_shape) > 3 else 1

print(f"Model input shape: {input_shape}")

# Label emotion 
class_names = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


PREDICTION_BUFFER_SIZE = 5  # Lưu 5 predictions gần nhất
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.4  

FRAME_SKIP = 2
frame_count = 0


def preprocess_face(frame, x, y, w, h):
   
    # Mở rộng vùng mặt 10% để capture đầy đủ hơn
    padding = int(0.1 * w)
    x_new = max(0, x - padding)
    y_new = max(0, y - padding)
    w_new = min(frame.shape[1] - x_new, w + 2 * padding)
    h_new = min(frame.shape[0] - y_new, h + 2 * padding)
    
    # Cắt vùng mặt với padding
    face_img = frame[y_new:y_new + h_new, x_new:x_new + w_new]
    
    # Cân bằng histogram trên ảnh grayscale để cải thiện độ tương phản
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.equalizeHist(face_gray)
    
    # Chuyển lại sang BGR rồi RGB
    face_bgr = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
    
    # Resize về kích thước model
    face_resized = cv2.resize(face_bgr, (img_width, img_height), 
                              interpolation=cv2.INTER_AREA)
    
    # Chuyển BGR → RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Preprocess cho MobileNetV2
    face_preprocessed = preprocess_input(face_rgb.astype("float32"))
    
    # Expand dims cho batch
    face_input = np.expand_dims(face_preprocessed, axis=0)
    return face_input

# =========================
# 5. HÀM SMOOTH PREDICTIONS
# =========================
def get_smoothed_prediction(preds):

    prediction_buffer.append(preds)
    
    if len(prediction_buffer) == 0:
        return preds
    
    # Tính trung bình của các predictions gần đây
    avg_preds = np.mean(prediction_buffer, axis=0)
    return avg_preds


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)            
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      

if not cap.isOpened():
    print("Không mở được webcam!")
    exit()

prev_time = 0
fps = 0.0
alpha_fps = 0.9

# Lưu prediction hiện tại để tránh nhấp nháy
current_emotion = "Detecting..."
current_confidence = 0.0

print("Nhấn 'q' để thoát.")
print("Nhấn 'r' để reset buffer.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame từ webcam.")
        break
    
    frame_count += 1


    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time
    if dt > 0:
        current_fps = 1.0 / dt
        fps = alpha_fps * fps + (1 - alpha_fps) * current_fps

    if frame_count % FRAME_SKIP == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
       
        gray = cv2.equalizeHist(gray)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,     
            minNeighbors=5,
            minSize=(80, 80),     
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            (x, y, w, h) = faces[0]

            
            face_input = preprocess_face(frame, x, y, w, h)
            preds = model.predict(face_input, verbose=0)[0]
            
            # Smooth predictions
            smoothed_preds = get_smoothed_prediction(preds)
            
            max_index = int(np.argmax(smoothed_preds))
            confidence = float(smoothed_preds[max_index])

            # Chỉ cập nhật nếu confidence đủ cao
            if confidence >= CONFIDENCE_THRESHOLD:
                if max_index < len(class_names):
                    current_emotion = class_names[max_index]
                else:
                    current_emotion = f"Class {max_index}"
                current_confidence = confidence
            else:
                current_emotion = "Uncertain"
                current_confidence = confidence

            
            if current_confidence >= 0.7:
                box_color = (0, 255, 0)  # Xanh lá: rất chắc chắn
            elif current_confidence >= 0.5:
                box_color = (0, 255, 255)  # Vàng: khá chắc chắn
            else:
                box_color = (0, 165, 255)  # Cam: không chắc chắn
            
            # Vẽ khung mặt
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Vẽ label với background để dễ đọc
            text = f"{current_emotion}: {current_confidence*100:.1f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background cho text
            cv2.rectangle(frame, 
                         (x, y - text_size[1] - 10), 
                         (x + text_size[0], y),
                         box_color, -1)
            
            # Text
            cv2.putText(frame, text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Hiển thị confidence bar
            bar_width = w
            bar_height = 10
            filled_width = int(bar_width * current_confidence)
            cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 5 + bar_height), 
                         (200, 200, 200), -1)
            cv2.rectangle(frame, (x, y + h + 5), (x + filled_width, y + h + 5 + bar_height), 
                         box_color, -1)
        else:
            # Không detect được mặt, clear buffer
            prediction_buffer.clear()
            current_emotion = "No face detected"
            current_confidence = 0.0

    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Buffer size
    cv2.putText(frame, f"Buffer: {len(prediction_buffer)}/{PREDICTION_BUFFER_SIZE}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Emotion Recognition - Optimized", frame)

  
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset buffer
        prediction_buffer.clear()
        print("Buffer reset!")


cap.release()
cv2.destroyAllWindows()
print("Đã đóng webcam.")