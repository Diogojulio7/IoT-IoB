
import cv2
import os
import numpy as np

dataset = "dataset"
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "lbph_model.yml")

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def load_dataset():
    faces, labels, label_map = [], [], {}
    label_id = 0

    for person in sorted(os.listdir(dataset)):
        person_path = os.path.join(dataset, person)
        if not os.path.isdir(person_path): continue

        label_map[label_id] = person
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            faces.append(img)
            labels.append(label_id)
        label_id += 1

    return faces, labels, label_map

# Treina modelo
faces, labels, label_map = load_dataset()
if len(faces) == 0:
    raise Exception("Dataset vazio. Colete imagens com collect_faces.py")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.write(model_path)
print("Modelo treinado e salvo em:", model_path)

# Reconhecimento em tempo real
recognizer.read(model_path)

cap = cv2.VideoCapture(0)
print("Pressione Q para sair")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200,200))

        label, confidence = recognizer.predict(face_resized)

        if confidence < 70:
            name = label_map[label]
            color = (0,255,0)
        else:
            name = "Desconhecido"
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{name} ({confidence:.1f})", (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
