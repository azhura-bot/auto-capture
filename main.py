import time
import math
from pathlib import Path

import cv2
import mediapipe as mp


# =========================
# KONFIGURASI
# =========================
POSE_LABELS = ["left", "right"]
TARGET_IMAGES_PER_CLASS = 200

# Countdown hanya saat pindah pose
POSE_CHANGE_COUNTDOWN_SECONDS = 5

DATASET_DIR = Path("dataset")
CAMERA_INDEX = 0
FORCE_ZOOM_OUT = True
# Nilai kecil biasanya lebih "wide" (tergantung driver kamera).
# Banyak webcam fixed-focus akan mengabaikan properti ini.
CAMERA_ZOOM_VALUE = 0
AUTO_TRY_ZOOM_VALUES = True
# Kandidat lintas driver: ada kamera yang wide di 0, ada yang pakai skala lain.
ZOOM_CANDIDATES = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

# Jeda antar auto-capture untuk pose yang sama
CAPTURE_DELAY_SECONDS = 0.25

# Kalau True, hanya simpan gambar jika pose terdeteksi
SAVE_ONLY_IF_POSE_DETECTED = True
REQUIRE_FULL_BODY_LANDMARKS = False
USE_POSE_RULE_FILTER = True

# Aturan sederhana validasi pose
LANDMARK_VISIBILITY_THRESHOLD = 0.55
IDLE_KNEE_ANGLE_MIN = 175.0
UP_KNEE_ANGLE_MIN = 178.0
DOWN_KNEE_ANGLE_MIN = 160.0
DOWN_KNEE_ANGLE_MAX = 173.0
ARM_RAISE_HIP_DELTA_Y_MIN = 0.03
ARM_WRIST_SIDE_GAP_MIN = 0.03
NON_ACTIVE_WRIST_MAX_RAISE = 0.01
REQUIRED_VALID_STREAK = 3
POSE_HOLD_MAX_MISSING_FRAMES = 6

# Ukuran preview
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
MIRROR_PREVIEW = True
SHOW_OVERLAY_TEXT = False


# =========================
# UTILITIES
# =========================
def ensure_directories():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for label in POSE_LABELS:
        (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)


def count_existing_images(label: str) -> int:
    folder = DATASET_DIR / label
    valid_ext = {".jpg", ".jpeg", ".png"}
    return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_ext)


def get_next_filename(label: str, index: int) -> Path:
    return DATASET_DIR / label / f"{label}_{index:04d}.jpg"


def draw_multiline_text(
    image,
    lines,
    start_x=20,
    start_y=35,
    line_spacing=35,
    color=(0, 255, 0),
    thickness=2,
    scale=0.9,
):
    y = start_y
    for line in lines:
        cv2.putText(
            image,
            line,
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_spacing


def save_frame(path: Path, frame):
    cv2.imwrite(str(path), frame)


def release_all(cap):
    cap.release()
    cv2.destroyAllWindows()


def landmark_visible(landmarks, idx, threshold=LANDMARK_VISIBILITY_THRESHOLD) -> bool:
    return landmarks[idx].visibility >= threshold


def calculate_angle(a, b, c) -> float:
    # Sudut ABC dalam derajat (0..180)
    ab_x, ab_y = a[0] - b[0], a[1] - b[1]
    cb_x, cb_y = c[0] - b[0], c[1] - b[1]
    dot = (ab_x * cb_x) + (ab_y * cb_y)
    mag_ab = math.sqrt((ab_x * ab_x) + (ab_y * ab_y))
    mag_cb = math.sqrt((cb_x * cb_x) + (cb_y * cb_y))
    if mag_ab == 0 or mag_cb == 0:
        return 180.0
    cosang = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cosang))


def evaluate_pose_for_label(pose_landmarks, mp_pose, label: str):
    if not pose_landmarks:
        return False, "Pose tidak terdeteksi", {}

    lm = pose_landmarks.landmark
    pl = mp_pose.PoseLandmark

    effective_label = label
    if MIRROR_PREVIEW:
        if label == "left":
            effective_label = "right"
        elif label == "right":
            effective_label = "left"

    if effective_label in ("left", "right"):
        required = [
            pl.LEFT_SHOULDER,
            pl.RIGHT_SHOULDER,
            pl.LEFT_ELBOW,
            pl.RIGHT_ELBOW,
            pl.LEFT_WRIST,
            pl.RIGHT_WRIST,
        ]
    else:
        required = [
            pl.LEFT_SHOULDER,
            pl.RIGHT_SHOULDER,
            pl.LEFT_HIP,
            pl.RIGHT_HIP,
            pl.LEFT_KNEE,
            pl.RIGHT_KNEE,
            pl.LEFT_ANKLE,
            pl.RIGHT_ANKLE,
        ]

    missing = [p.name for p in required if not landmark_visible(lm, p.value)]
    if missing:
        if effective_label in ("left", "right"):
            return False, "Upper body belum terlihat jelas (bahu-siku-pergelangan)", {}
        if REQUIRE_FULL_BODY_LANDMARKS:
            return False, "Full body belum terlihat (bahu-ankle)", {}

    left_knee = 180.0
    right_knee = 180.0
    avg_knee = 180.0
    if effective_label not in ("left", "right"):
        left_knee = calculate_angle(
            (lm[pl.LEFT_HIP.value].x, lm[pl.LEFT_HIP.value].y),
            (lm[pl.LEFT_KNEE.value].x, lm[pl.LEFT_KNEE.value].y),
            (lm[pl.LEFT_ANKLE.value].x, lm[pl.LEFT_ANKLE.value].y),
        )
        right_knee = calculate_angle(
            (lm[pl.RIGHT_HIP.value].x, lm[pl.RIGHT_HIP.value].y),
            (lm[pl.RIGHT_KNEE.value].x, lm[pl.RIGHT_KNEE.value].y),
            (lm[pl.RIGHT_ANKLE.value].x, lm[pl.RIGHT_ANKLE.value].y),
        )
        avg_knee = (left_knee + right_knee) / 2.0

    left_arm_raise_delta_y = lm[pl.LEFT_HIP.value].y - lm[pl.LEFT_WRIST.value].y
    right_arm_raise_delta_y = lm[pl.RIGHT_HIP.value].y - lm[pl.RIGHT_WRIST.value].y
    wrist_height_gap = lm[pl.RIGHT_WRIST.value].y - lm[pl.LEFT_WRIST.value].y

    if not USE_POSE_RULE_FILTER:
        return True, "Pose terdeteksi", {
            "left_knee": left_knee,
            "right_knee": right_knee,
            "avg_knee": avg_knee,
            "left_arm_raise_delta_y": left_arm_raise_delta_y,
            "right_arm_raise_delta_y": right_arm_raise_delta_y,
            "wrist_height_gap": wrist_height_gap,
        }

    if label == "idle":
        ok = avg_knee >= IDLE_KNEE_ANGLE_MIN
        msg = "Idle valid" if ok else "Idle gagal: lutut terlalu tekuk"
    elif label == "up":
        ok = avg_knee >= UP_KNEE_ANGLE_MIN
        msg = "Up valid" if ok else "Up gagal: luruskan lutut"
    elif label == "down":
        ok = DOWN_KNEE_ANGLE_MIN <= avg_knee <= DOWN_KNEE_ANGLE_MAX
        msg = "Down valid" if ok else "Down gagal: tekuk lutut lebih dalam"
    elif effective_label == "left":
        left_raised = left_arm_raise_delta_y >= ARM_RAISE_HIP_DELTA_Y_MIN
        higher_than_right = wrist_height_gap >= ARM_WRIST_SIDE_GAP_MIN
        right_not_raised = right_arm_raise_delta_y <= NON_ACTIVE_WRIST_MAX_RAISE
        ok = left_raised and higher_than_right and right_not_raised
        msg = (
            f"{label.capitalize()} valid"
            if ok
            else f"{label.capitalize()} gagal: tangan target harus di atas pinggang, tangan satunya tetap netral"
        )
    elif effective_label == "right":
        right_raised = right_arm_raise_delta_y >= ARM_RAISE_HIP_DELTA_Y_MIN
        higher_than_left = (-wrist_height_gap) >= ARM_WRIST_SIDE_GAP_MIN
        left_not_raised = left_arm_raise_delta_y <= NON_ACTIVE_WRIST_MAX_RAISE
        ok = right_raised and higher_than_left and left_not_raised
        msg = (
            f"{label.capitalize()} valid"
            if ok
            else f"{label.capitalize()} gagal: tangan target harus di atas pinggang, tangan satunya tetap netral"
        )
    else:
        ok = True
        msg = "Pose valid"

    return ok, msg, {
        "left_knee": left_knee,
        "right_knee": right_knee,
        "avg_knee": avg_knee,
        "left_arm_raise_delta_y": left_arm_raise_delta_y,
        "right_arm_raise_delta_y": right_arm_raise_delta_y,
        "wrist_height_gap": wrist_height_gap,
    }


def main():
    ensure_directories()

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Disarankan untuk Windows:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak bisa dibuka. Cek CAMERA_INDEX atau izin kamera.")

    if FORCE_ZOOM_OUT:
        if AUTO_TRY_ZOOM_VALUES:
            best_value = None
            best_readback = float("inf")
            accepted = []
            for z in ZOOM_CANDIDATES:
                ok = cap.set(cv2.CAP_PROP_ZOOM, z)
                rb = cap.get(cv2.CAP_PROP_ZOOM)
                accepted.append((z, ok, rb))
                # Asumsi umum: readback lebih kecil cenderung lebih wide.
                if ok and rb < best_readback:
                    best_readback = rb
                    best_value = z

            if best_value is not None:
                cap.set(cv2.CAP_PROP_ZOOM, best_value)
                final_rb = cap.get(cv2.CAP_PROP_ZOOM)
                print(f"[KAMERA] Auto zoom-out pilih value={best_value}, readback={final_rb:.2f}")
            else:
                print("[KAMERA] Kamera tidak menerima perubahan zoom (kemungkinan fixed lens).")

            print("[KAMERA] Hasil uji zoom:")
            for z, ok, rb in accepted:
                print(f"  - try={z:>3} -> {'OK' if ok else 'NO'} | readback={rb:.2f}")
        else:
            zoom_ok = cap.set(cv2.CAP_PROP_ZOOM, CAMERA_ZOOM_VALUE)
            current_zoom = cap.get(cv2.CAP_PROP_ZOOM)
            print(
                f"[KAMERA] Set zoom={CAMERA_ZOOM_VALUE} -> "
                f"{'berhasil' if zoom_ok else 'gagal/diabaikan'}, baca balik={current_zoom:.2f}"
            )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Tekan 'q' untuk keluar kapan saja.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) as pose:
        for label in POSE_LABELS:
            existing_count = count_existing_images(label)
            last_pose_landmarks = None
            missing_pose_frames = 0

            if existing_count >= TARGET_IMAGES_PER_CLASS:
                print(f"[SKIP] Kelas '{label}' sudah memiliki {existing_count} gambar.")
                continue

            print(f"\n=== Sekarang pose: {label.upper()} ===")
            print(f"Target: {TARGET_IMAGES_PER_CLASS} gambar")

            # =========================================
            # 1) COUNTDOWN HANYA SEKALI SAAT MASUK POSE
            # =========================================
            countdown_start = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Gagal membaca frame dari webcam.")
                    release_all(cap)
                    return

                base_frame = cv2.flip(frame, 1)
                dataset_frame = base_frame.copy()
                preview_frame = dataset_frame.copy()
                rgb = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if results.pose_landmarks:
                    last_pose_landmarks = results.pose_landmarks
                    missing_pose_frames = 0
                    effective_pose_landmarks = results.pose_landmarks
                else:
                    missing_pose_frames += 1
                    if missing_pose_frames <= POSE_HOLD_MAX_MISSING_FRAMES:
                        effective_pose_landmarks = last_pose_landmarks
                    else:
                        effective_pose_landmarks = None
                        last_pose_landmarks = None

                if effective_pose_landmarks:
                    mp_drawing.draw_landmarks(
                        dataset_frame,
                        effective_pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                    mp_drawing.draw_landmarks(
                        preview_frame,
                        effective_pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                pose_valid, pose_msg, pose_meta = evaluate_pose_for_label(
                    effective_pose_landmarks, mp_pose, label
                )
                if pose_meta:
                    if label in ("left", "right"):
                        metric_text = (
                            f"Raise(HIP) L/R: {pose_meta['left_arm_raise_delta_y']:+.3f}/"
                            f"{pose_meta['right_arm_raise_delta_y']:+.3f} | "
                            f"Gap R-L: {pose_meta['wrist_height_gap']:+.3f}"
                        )
                    else:
                        metric_text = f"Lutut L/R: {pose_meta['left_knee']:.0f}/{pose_meta['right_knee']:.0f}"
                else:
                    metric_text = "Metric: -"

                elapsed = time.time() - countdown_start
                remaining = max(
                    0,
                    int(POSE_CHANGE_COUNTDOWN_SECONDS - elapsed) + (0 if elapsed.is_integer() else 1)
                )

                info_lines = [
                    f"Sekarang: {label}",
                    f"Progress: {existing_count}/{TARGET_IMAGES_PER_CLASS}",
                    metric_text,
                    f"Status pose: {pose_msg}",
                    f"Mulai capture dalam: {remaining} detik",
                    "Tekan 's' untuk buka setting kamera",
                    "Tahan pose, lalu auto-capture akan berjalan cepat",
                    "Tekan 'q' untuk keluar",
                ]
                if SHOW_OVERLAY_TEXT:
                    draw_multiline_text(preview_frame, info_lines, color=(0, 255, 255))

                cv2.imshow("Pose Dataset Collector", preview_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Program dihentikan oleh user.")
                    release_all(cap)
                    return
                if key == ord("s"):
                    cap.set(cv2.CAP_PROP_SETTINGS, 1)

                if elapsed >= POSE_CHANGE_COUNTDOWN_SECONDS:
                    break

            # =========================================
            # 2) AUTO-CAPTURE CEPAT UNTUK POSE YANG SAMA
            # =========================================
            last_capture_time = 0.0
            valid_streak = 0

            while existing_count < TARGET_IMAGES_PER_CLASS:
                ret, frame = cap.read()
                if not ret:
                    print("Gagal membaca frame dari webcam.")
                    release_all(cap)
                    return

                base_frame = cv2.flip(frame, 1)
                dataset_frame = base_frame.copy()
                preview_frame = dataset_frame.copy()
                rgb = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if results.pose_landmarks:
                    last_pose_landmarks = results.pose_landmarks
                    missing_pose_frames = 0
                    effective_pose_landmarks = results.pose_landmarks
                else:
                    missing_pose_frames += 1
                    if missing_pose_frames <= POSE_HOLD_MAX_MISSING_FRAMES:
                        effective_pose_landmarks = last_pose_landmarks
                    else:
                        effective_pose_landmarks = None
                        last_pose_landmarks = None

                if effective_pose_landmarks:
                    mp_drawing.draw_landmarks(
                        dataset_frame,
                        effective_pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                    mp_drawing.draw_landmarks(
                        preview_frame,
                        effective_pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                pose_valid, pose_msg, pose_meta = evaluate_pose_for_label(
                    effective_pose_landmarks, mp_pose, label
                )
                if pose_meta:
                    if label in ("left", "right"):
                        metric_text = (
                            f"Raise(HIP) L/R: {pose_meta['left_arm_raise_delta_y']:+.3f}/"
                            f"{pose_meta['right_arm_raise_delta_y']:+.3f} | "
                            f"Gap R-L: {pose_meta['wrist_height_gap']:+.3f}"
                        )
                    else:
                        metric_text = f"Lutut L/R: {pose_meta['left_knee']:.0f}/{pose_meta['right_knee']:.0f}"
                else:
                    metric_text = "Metric: -"

                now = time.time()
                time_until_next_capture = max(0.0, CAPTURE_DELAY_SECONDS - (now - last_capture_time))

                info_lines = [
                    f"Sekarang: {label}",
                    f"Progress: {existing_count}/{TARGET_IMAGES_PER_CLASS}",
                    metric_text,
                    f"Status pose: {pose_msg}",
                    f"Capture berikutnya: {time_until_next_capture:.2f} detik",
                    "Tekan 's' untuk buka setting kamera",
                    "Tekan 'n' untuk lanjut ke pose berikutnya",
                    "Tekan 'q' untuk keluar",
                ]
                if SHOW_OVERLAY_TEXT:
                    draw_multiline_text(preview_frame, info_lines, color=(0, 255, 0))

                cv2.imshow("Pose Dataset Collector", preview_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Program dihentikan oleh user.")
                    release_all(cap)
                    return
                if key == ord("s"):
                    cap.set(cv2.CAP_PROP_SETTINGS, 1)

                # opsional: tekan n untuk langsung lanjut ke pose berikutnya
                if key == ord("n"):
                    print(f"[NEXT] Pindah dari pose '{label}' dengan total {existing_count} gambar.")
                    break

                if now - last_capture_time < CAPTURE_DELAY_SECONDS:
                    continue

                pose_detected = effective_pose_landmarks is not None
                if SAVE_ONLY_IF_POSE_DETECTED and not pose_detected:
                    valid_streak = 0
                    continue
                if not pose_valid:
                    valid_streak = 0
                    continue

                valid_streak += 1
                if valid_streak < REQUIRED_VALID_STREAK:
                    continue

                file_path = get_next_filename(label, existing_count + 1)
                save_frame(file_path, dataset_frame)
                existing_count += 1
                last_capture_time = now
                valid_streak = 0

                print(f"[{label}] Tersimpan: {file_path.name} ({existing_count}/{TARGET_IMAGES_PER_CLASS})")

                # flash singkat
                flash_start = time.time()
                while time.time() - flash_start < 0.25:
                    flash_frame = preview_frame.copy()
                    if SHOW_OVERLAY_TEXT:
                        draw_multiline_text(
                            flash_frame,
                            [
                                f"Sekarang: {label}",
                                f"Progress: {existing_count}/{TARGET_IMAGES_PER_CLASS}",
                                "CAPTURE BERHASIL",
                            ],
                            color=(0, 255, 0),
                        )
                    cv2.imshow("Pose Dataset Collector", flash_frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        print("Program dihentikan oleh user.")
                        release_all(cap)
                        return

            print(f"[SELESAI] Kelas '{label}' lengkap: {existing_count} gambar.")

        print("\nSemua dataset selesai dikumpulkan.")
        release_all(cap)


if __name__ == "__main__":
    main()
