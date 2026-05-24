import cv2
import pandas as pd
import numpy as np
import math
from ultralytics import YOLO
import os
from collections import defaultdict

# ścieżki dostępu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset_stereo")

LEFT_DIR = os.path.join(DATASET_DIR, "left")
RIGHT_DIR = os.path.join(DATASET_DIR, "right")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "output_prototype2.mp4")
GROUND_TRUTH_CSV = os.path.join(DATASET_DIR, "ground_truth.csv")

# Ładowanie modelu yolo
model = YOLO('yolov8m.pt')

# Parametry kamery (z generate_data.py / IMX490)
WIDTH    = 2896
HEIGHT   = 1876
FOV_DEG  = 120
BASELINE = 0.2  # metry

# Ogniskowa w pikselach:  f = (W/2) / tan(FOV/2)
FOCAL_PX = (WIDTH / 2) / math.tan(math.radians(FOV_DEG / 2))  # ≈ 836 px

# ============================================================
# Parametry stabilizacji
# ============================================================
EMA_ALPHA = 0.3          # Waga nowego pomiaru w EMA (0.0–1.0). Mniejsza wartość = gładsze
OUTLIER_THRESHOLD = 0.30 # Pomiary odchylone o >30% od EMA są tłumione
OUTLIER_ALPHA = 0.05     # Waga outlieru – prawie go ignorujemy
MIN_DISPARITY = 1.0      # Minimalna dozwolona dysparycja (piksel)
IOU_THRESHOLD = 0.3      # Próg IoU do śledzenia obiektów między klatkami


def get_center_float(bbox):
    """Zwraca środek bounding boxa jako float (x_center, y_center) – bez zaokrąglania."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_iou(boxA, boxB):
    """Oblicza IoU (Intersection over Union) między dwoma bounding boxami [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def match_objects(boxes_left, boxes_right):
    """
    Dopasowanie obiektów lewo-prawo na podstawie:
    - zgodności klasy
    - podobnej pozycji Y (epipolar constraint)
    - podobnego rozmiaru bounding boxa (stosunek szerokości i wysokości)
    """
    matches = []
    used_right = set()

    for boxL in boxes_left:
        best_match = None
        best_score = float('inf')

        cL = get_center_float(boxL[:4])
        wL = boxL[2] - boxL[0]
        hL = boxL[3] - boxL[1]

        for j, boxR in enumerate(boxes_right):
            if j in used_right:
                continue

            cR = get_center_float(boxR[:4])
            wR = boxR[2] - boxR[0]
            hR = boxR[3] - boxR[1]

            y_diff = abs(cL[1] - cR[1])

            # Warunki: ta sama klasa, podobna wysokość Y, lewy obiekt musi być bardziej na prawo
            if boxL[5] != boxR[5]:
                continue
            if y_diff > 30:
                continue

            # Dodatkowy warunek: podobny rozmiar bounding boxa (±50%)
            w_ratio = wL / max(wR, 1)
            h_ratio = hL / max(hR, 1)
            if w_ratio < 0.5 or w_ratio > 2.0 or h_ratio < 0.5 or h_ratio > 2.0:
                continue

            # Wynik: mniejsza różnica Y + mniejsza różnica rozmiaru = lepsze dopasowanie
            size_diff = abs(wL - wR) + abs(hL - hR)
            score = y_diff + size_diff * 0.1

            if score < best_score:
                best_score = score
                best_match = (j, boxR)

        if best_match is not None:
            used_right.add(best_match[0])
            matches.append((boxL, best_match[1]))

    return matches


class ObjectTracker:
    """
    Prosty tracker obiektów oparty na IoU.
    Przypisuje stałe ID obiektom i utrzymuje historię odległości (EMA).
    """

    def __init__(self):
        self.next_id = 0
        self.tracks = {}  # track_id -> {'bbox': [...], 'ema_distance': float, 'age': int, 'missed': int}

    def update(self, detections):
        """
        detections: lista dict {'bbox': [x1,y1,x2,y2], 'class': int, 'distance': float}
        Zwraca: lista dict z dodanym 'track_id' i 'smoothed_distance'
        """
        results = []

        if not self.tracks:
            # Pierwsze klatki – inicjalizujemy wszystkie tracki
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'ema_distance': det['distance'],
                    'age': 1,
                    'missed': 0,
                }
                results.append({
                    **det,
                    'track_id': tid,
                    'smoothed_distance': det['distance'],
                })
            return results

        # Dopasuj detekcje do istniejących tracków na podstawie IoU
        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        # Oblicz macierz IoU
        iou_matrix = []
        for tid in track_ids:
            row = []
            for di, det in enumerate(detections):
                if self.tracks[tid]['class'] == det['class']:
                    iou = compute_iou(self.tracks[tid]['bbox'], det['bbox'])
                else:
                    iou = 0.0
                row.append(iou)
            iou_matrix.append(row)

        # Greedy matching: najlepsze IoU najpierw
        if iou_matrix and detections:
            iou_arr = np.array(iou_matrix)
            while True:
                if iou_arr.size == 0:
                    break
                max_iou = iou_arr.max()
                if max_iou < IOU_THRESHOLD:
                    break
                ti, di = np.unravel_index(iou_arr.argmax(), iou_arr.shape)

                tid = track_ids[ti]
                det = detections[di]

                # Aktualizacja EMA z tłumieniem outlierów
                old_ema = self.tracks[tid]['ema_distance']
                new_dist = det['distance']

                if old_ema > 0:
                    relative_change = abs(new_dist - old_ema) / old_ema
                    if relative_change > OUTLIER_THRESHOLD:
                        alpha = OUTLIER_ALPHA  # outlier – prawie ignoruj
                    else:
                        alpha = EMA_ALPHA
                else:
                    alpha = 1.0

                smoothed = alpha * new_dist + (1 - alpha) * old_ema

                self.tracks[tid]['bbox'] = det['bbox']
                self.tracks[tid]['ema_distance'] = smoothed
                self.tracks[tid]['age'] += 1
                self.tracks[tid]['missed'] = 0

                results.append({
                    **det,
                    'track_id': tid,
                    'smoothed_distance': smoothed,
                })

                matched_tracks.add(ti)
                matched_dets.add(di)

                # Wyzeruj wiersz i kolumnę, żeby nie dopasować ponownie
                iou_arr[ti, :] = -1
                iou_arr[:, di] = -1

        # Nowe detekcje (nie dopasowane do żadnego tracka)
        for di, det in enumerate(detections):
            if di not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'ema_distance': det['distance'],
                    'age': 1,
                    'missed': 0,
                }
                results.append({
                    **det,
                    'track_id': tid,
                    'smoothed_distance': det['distance'],
                })

        # Zwiększ 'missed' dla nie-dopasowanych tracków, usuń stare
        for ti, tid in enumerate(track_ids):
            if ti not in matched_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > 10:
                    del self.tracks[tid]

        return results


def compute_subpixel_disparity(boxL, boxR):
    """
    Oblicza dysparycję na podstawie wielu punktów bounding boxa
    (centrum, lewy brzeg, prawy brzeg) i uśrednia — daje bardziej
    stabilny wynik niż samo centrum.
    """
    # Środki (float)
    cx_L = (boxL[0] + boxL[2]) / 2.0
    cx_R = (boxR[0] + boxR[2]) / 2.0
    disp_center = cx_L - cx_R

    # Lewe krawędzie
    disp_left = boxL[0] - boxR[0]

    # Prawe krawędzie
    disp_right = boxL[2] - boxR[2]

    # Uśrednienie 3 pomiarów dysparycji (mediana jest odporna na outlier)
    disparities = [disp_center, disp_left, disp_right]
    return float(np.median(disparities))


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

    # Tracker obiektów
    tracker = ObjectTracker()

    for i, frame_name in enumerate(frames):
        img_L_path = os.path.join(LEFT_DIR, frame_name)
        img_R_path = os.path.join(RIGHT_DIR, frame_name)

        img_L = cv2.imread(img_L_path)
        img_R = cv2.imread(img_R_path)

        # Zabezpieczenie przed uszkodzonymi klatkaki
        if img_L is None or img_R is None:
            print(f"uszkodzony plik: {frame_name}. Pomijam tę klatkę.")
            continue

        # Nakładanie maski na maskę samochodu - zostawiamy górne 80% obrazu
        h, w = img_L.shape[:2]
        cutoff_y = int(h * 0.80) # można dostosować, jeśli odcina za dużo/za mało

        # Tworzymy kopie obrazów do detekcji, aby nie mieć czarnego paska na gotowym wideo
        img_L_det = img_L.copy()
        img_R_det = img_R.copy()

        # Zamazujemy dolną część na czarno
        img_L_det[cutoff_y:, :] = 0
        img_R_det[cutoff_y:, :] = 0
        # -----------------------------------------------------

        # 1. Detekcja YOLO na obu kamerach (używamy obrazów z MASKĄ)
        results_L = model.predict(img_L_det, verbose=False, classes=[2, 3, 5, 7]) 
        results_R = model.predict(img_R_det, verbose=False, classes=[2, 3, 5, 7])

        boxes_L = results_L[0].boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, class]
        boxes_R = results_R[0].boxes.data.cpu().numpy()

        # 2. Dopasowanie obiektów z lewej i prawej kamery
        matched_pairs = match_objects(boxes_L, boxes_R)

        # 3. Obliczanie surowej odległości
        raw_detections = []
        for boxL, boxR in matched_pairs:
            # Sub-pikselowa dysparycja (mediana z 3 pomiarów)
            disparity = compute_subpixel_disparity(boxL, boxR)

            # Zabezpieczenie przed dysparycją bliską zeru
            if disparity >= MIN_DISPARITY:
                distance = (FOCAL_PX * BASELINE) / disparity
                raw_detections.append({
                    'bbox': list(map(float, boxL[:4])),
                    'class': int(boxL[5]),
                    'conf': float(boxL[4]),
                    'distance': distance,
                })

        # 4. Aktualizacja trackera i wygładzanie
        tracked = tracker.update(raw_detections)

        # 5. Rysowanie na obrazie z lewej kamery
        for obj in tracked:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            smoothed_dist = obj['smoothed_distance']

            cv2.rectangle(img_L, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Dist: {smoothed_dist:.2f}m"
            cv2.putText(img_L, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 6. Zapis klatki do wideo
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
