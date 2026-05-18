import cv2
import pandas as pd
import numpy as np
import math
from ultralytics import YOLO
import os

# --- KONFIGURACJA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset_stereo")

LEFT_DIR = os.path.join(DATASET_DIR, "left")
RIGHT_DIR = os.path.join(DATASET_DIR, "right")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "output_prototype.mp4")
GROUND_TRUTH_CSV = os.path.join(DATASET_DIR, "ground_truth.csv")

# Ładowanie modelu YOLOv8 (wersja 'm' to dobry kompromis między szybkością a dokładnością)
model = YOLO('yolov8m.pt')

# Parametry kamery (z generate_data.py / IMX490)
WIDTH    = 2896
HEIGHT   = 1876
FOV_DEG  = 120
BASELINE = 0.2  # metry

# Ogniskowa w pikselach:  f = (W/2) / tan(FOV/2)
FOCAL_PX = (WIDTH / 2) / math.tan(math.radians(FOV_DEG / 2))  # ≈ 836 px

def get_center(bbox):
    """Zwraca środek bounding boxa (x_center, y_center)"""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def match_objects(boxes_left, boxes_right):
    """
    Proste dopasowanie: zakłada, że obiekty na podobnej wysokości (y) 
    i podobnej klasie to ten sam pojazd.
    """
    matches = []
    for boxL in boxes_left:
        best_match = None
        min_y_diff = float('inf')
        
        cL = get_center(boxL[:4])
        
        for boxR in boxes_right:
            cR = get_center(boxR[:4])
            y_diff = abs(cL[1] - cR[1]) # Różnica w osi Y powinna być bliska 0
            
            # Jeśli klasy się zgadzają i są na podobnej wysokości (tolerancja np. 30 pikseli)
            if boxL[5] == boxR[5] and y_diff < 30 and y_diff < min_y_diff:
                min_y_diff = y_diff
                best_match = boxR
                
        if best_match is not None:
            matches.append((boxL, best_match))
    return matches

def main():
    # Pobranie listy plików (zakładamy synchroniczne nazwy)
    frames = sorted([f for f in os.listdir(LEFT_DIR) if f.endswith('.jpg')])
    total_frames = len(frames)
    print(f"Rozpoczynam przetwarzanie {total_frames} klatek...")
    
    # Konfiguracja zapisywacza wideo (VideoWriter)
    sample_frame = cv2.imread(os.path.join(LEFT_DIR, frames[0]))
    h, w, _ = sample_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 10.0, (w, h))
    
    # Opcjonalnie: wczytaj CSV by nakładać Ground Truth na wideo
    # gt_data = pd.read_csv(GROUND_TRUTH_CSV)

    for i, frame_name in enumerate(frames):
        img_L_path = os.path.join(LEFT_DIR, frame_name)
        img_R_path = os.path.join(RIGHT_DIR, frame_name)
        
        img_L = cv2.imread(img_L_path)
        img_R = cv2.imread(img_R_path)
        
        # 1. Detekcja YOLO na obu kamerach
        results_L = model.predict(img_L, verbose=False, classes=[2, 3, 5, 7]) # Detekcja pojazdów (COCO)
        results_R = model.predict(img_R, verbose=False, classes=[2, 3, 5, 7])
        
        boxes_L = results_L[0].boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, class]
        boxes_R = results_R[0].boxes.data.cpu().numpy()
        
        # 2. Dopasowanie obiektów z lewej i prawej kamery
        matched_pairs = match_objects(boxes_L, boxes_R)
        
        # 3. Obliczanie odległości i rysowanie na obrazie z lewej kamery
        for boxL, boxR in matched_pairs:
            x_center_L = get_center(boxL[:4])[0]
            x_center_R = get_center(boxR[:4])[0]
            
            # Dysparycja (różnica w pikselach w osi X)
            disparity = x_center_L - x_center_R
            
            # Zabezpieczenie przed dzieleniem przez zero
            if disparity > 0:
                # Obliczenie dystansu: Z = (f · B) / d
                distance = (FOCAL_PX * BASELINE) / disparity
                
                # Rysowanie bounding boxa i odległości
                x1, y1, x2, y2 = map(int, boxL[:4])
                cv2.rectangle(img_L, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"Dist: {distance:.2f}m"
                cv2.putText(img_L, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 4. Zapis klatki do wideo
        out.write(img_L)
        
        if i % 10 == 0:
            print(f"Postęp: {i}/{total_frames} klatek")
        
        # (Opcjonalnie) Podgląd na żywo:
        # cv2.imshow('Autonomiczny Prototyp', cv2.resize(img_L, (1024, 768)))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    out.release()
    cv2.destroyAllWindows()
    print(f"Zapisano gotowe wideo: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
