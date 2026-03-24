import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# USER SETTINGS
# -----------------------------
REAL_HEIGHT_CM = 165  # change this

# -----------------------------
# LOAD MODEL
# -----------------------------
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

pose = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

calibrated = False
SCALE = 1

print("➡️ Stand straight for calibration...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = pose.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    status_text = "Detecting..."
    color = (0, 255, 255)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        head = landmarks[0]
        left_foot = landmarks[27]
        right_foot = landmarks[28]

        h, w, _ = frame.shape

        head_y = int(head.y * h)
        foot_y = int(max(left_foot.y, right_foot.y) * h)

        pixel_height = abs(foot_y - head_y)

        # -----------------------------
        # DISTANCE CHECK
        # -----------------------------
        if pixel_height < 250:
            status_text = "Move Closer"
            color = (0, 0, 255)

        elif pixel_height > 600:
            status_text = "Move Back"
            color = (0, 0, 255)

        else:
            # -----------------------------
            # CALIBRATION
            # -----------------------------
            if not calibrated:
                SCALE = REAL_HEIGHT_CM / pixel_height
                calibrated = True
                print(f"✅ Calibration Done! Scale = {SCALE:.4f}")

            # -----------------------------
            # HEIGHT CALCULATION
            # -----------------------------
            height_cm = pixel_height * SCALE

            status_text = f"Height: {int(height_cm)} cm"
            color = (0, 255, 0)

            # Draw measurement line
            cv2.line(frame, (100, head_y), (100, foot_y), (0, 255, 0), 3)

    else:
        status_text = "Show Full Body"
        color = (0, 0, 255)

    # -----------------------------
    # UI BOX
    # -----------------------------
    cv2.rectangle(frame, (20, 20), (350, 90), (0, 0, 0), -1)

    cv2.putText(frame, status_text,
                (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2)

    cv2.imshow("AI Height Estimator PRO", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()