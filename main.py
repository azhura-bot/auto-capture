import time
from pathlib import Path

import cv2
import mediapipe as mp


# =========================
# KONFIGURASI
# =========================
POSE_LABELS = ["left", "right", "up", "down", "idle"]
TARGET_IMAGES_PER_CLASS = 200

# Countdown hanya saat pindah pose
POSE_CHANGE_COUNTDOWN_SECONDS = 5

DATASET_DIR = Path("dataset_raw")
CAMERA_INDEX = 0

# Jeda antar auto-capture untuk pose yang sama
CAPTURE_DELAY_SECONDS = 0.25

# Kalau True, hanya simpan gambar jika pose terdeteksi
SAVE_ONLY_IF_POSE_DETECTED = True

# Ukuran preview
FRAME_WIDTH = 960
FRAME_HEIGHT = 720


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


def main():
    ensure_directories()

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Disarankan untuk Windows:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak bisa dibuka. Cek CAMERA_INDEX atau izin kamera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Tekan 'q' untuk keluar kapan saja.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for label in POSE_LABELS:
            existing_count = count_existing_images(label)

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

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                elapsed = time.time() - countdown_start
                remaining = max(
                    0,
                    int(POSE_CHANGE_COUNTDOWN_SECONDS - elapsed) + (0 if elapsed.is_integer() else 1)
                )

                info_lines = [
                    f"Sekarang: {label}",
                    f"Progress: {existing_count}/{TARGET_IMAGES_PER_CLASS}",
                    f"Mulai capture dalam: {remaining} detik",
                    "Tahan pose, lalu auto-capture akan berjalan cepat",
                    "Tekan 'q' untuk keluar",
                ]
                draw_multiline_text(frame, info_lines, color=(0, 255, 255))

                cv2.imshow("Pose Dataset Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Program dihentikan oleh user.")
                    release_all(cap)
                    return

                if elapsed >= POSE_CHANGE_COUNTDOWN_SECONDS:
                    break

            # =========================================
            # 2) AUTO-CAPTURE CEPAT UNTUK POSE YANG SAMA
            # =========================================
            last_capture_time = 0.0

            while existing_count < TARGET_IMAGES_PER_CLASS:
                ret, frame = cap.read()
                if not ret:
                    print("Gagal membaca frame dari webcam.")
                    release_all(cap)
                    return

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                now = time.time()
                time_until_next_capture = max(0.0, CAPTURE_DELAY_SECONDS - (now - last_capture_time))

                info_lines = [
                    f"Sekarang: {label}",
                    f"Progress: {existing_count}/{TARGET_IMAGES_PER_CLASS}",
                    f"Capture berikutnya: {time_until_next_capture:.2f} detik",
                    "Tekan 'n' untuk lanjut ke pose berikutnya",
                    "Tekan 'q' untuk keluar",
                ]
                draw_multiline_text(frame, info_lines, color=(0, 255, 0))

                cv2.imshow("Pose Dataset Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Program dihentikan oleh user.")
                    release_all(cap)
                    return

                # opsional: tekan n untuk langsung lanjut ke pose berikutnya
                if key == ord("n"):
                    print(f"[NEXT] Pindah dari pose '{label}' dengan total {existing_count} gambar.")
                    break

                if now - last_capture_time < CAPTURE_DELAY_SECONDS:
                    continue

                pose_detected = results.pose_landmarks is not None
                if SAVE_ONLY_IF_POSE_DETECTED and not pose_detected:
                    continue

                file_path = get_next_filename(label, existing_count + 1)
                save_frame(file_path, frame)
                existing_count += 1
                last_capture_time = now

                print(f"[{label}] Tersimpan: {file_path.name} ({existing_count}/{TARGET_IMAGES_PER_CLASS})")

                # flash singkat
                flash_start = time.time()
                while time.time() - flash_start < 0.25:
                    flash_frame = frame.copy()
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