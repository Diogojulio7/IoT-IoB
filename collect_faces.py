
import cv2
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help='Nome da pessoa a ser coletada')
parser.add_argument('--num', type=int, default=100, help='Número de imagens')
parser.add_argument('--output', default='dataset', help='Diretório de saída')
args = parser.parse_args()

out_dir = os.path.join(args.output, args.name)
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
collected = 0
print(f"Iniciando coleta para '{args.name}' — pressione Q para sair.")

# Usa o Haar Cascade nativo do OpenCV
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

while collected < args.num:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200,200))
        ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = os.path.join(out_dir, f"{args.name}_{ts}.jpg")
        cv2.imwrite(filename, face_resized)
        collected += 1

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{collected}/{args.num}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        break

    cv2.imshow("Coleta de Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("Coleta finalizada!")
