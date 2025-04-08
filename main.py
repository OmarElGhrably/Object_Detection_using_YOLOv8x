from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')

# We're only interested in these classes
TARGET_CLASSES = ['person', 'bottle', 'cup', 'vase', 'wine glass', 'bowl']

# Function to check if two boxes are close to each other
def is_near(box1, box2, threshold=0.3):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    dx = min(x2, x4) - max(x1, x3)
    dy = min(y2, y4) - max(y1, y3)
    if dx >= 0 and dy >= 0:
        overlap_area = dx * dy
        area1 = (x2 - x1) * (y2 - y1)
        return overlap_area / area1 > threshold
    return False

# Process a single image
def process_image(image_path, save_path="output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Image not found.")
        return

    print(f"Image shape: {img.shape}")
    results = model.predict(source=img, conf=0.25)
    boxes = []

    print(f"Detected {len(results[0].boxes)} objects.")
    for box in results[0].boxes:
        cls_id = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        label = model.names[cls_id]

        if label in TARGET_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            boxes.append({'class': label, 'box': [x1, y1, x2, y2], 'confidence': conf})

    persons = [obj for obj in boxes if obj['class'] == 'person']
    objects = [obj for obj in boxes if obj['class'] in TARGET_CLASSES and obj['class'] != 'person']

    for obj in objects:
        x1, y1, x2, y2 = obj['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{obj['class']} {obj['confidence']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for person in persons:
        is_holding = False
        px1, py1, px2, py2 = person['box']

        for obj in objects:
            if is_near(person['box'], obj['box']):
                is_holding = True
                ox1, oy1, ox2, oy2 = obj['box']
                cv2.rectangle(img, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                cv2.putText(img, obj['class'], (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ✅ Draw person box after checking
        if is_holding:
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(img, "Person is holding something", (px1, py1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(img, "Person", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"Total objects detected and annotated: {len(boxes)}")
    cv2.imwrite(save_path, img)
    print(f"[INFO] Saved output image: {save_path}")
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process a video file
def process_video(video_path, output_path='output_video.avi'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Unable to open video file.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_skip = 3
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        results = model.predict(source=frame, conf=0.25)
        boxes = []

        for box in results[0].boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            label = model.names[cls_id]

            if label in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                boxes.append({'class': label, 'box': [x1, y1, x2, y2], 'confidence': conf})

        persons = [obj for obj in boxes if obj['class'] == 'person']
        objects = [obj for obj in boxes if obj['class'] in TARGET_CLASSES and obj['class'] != 'person']

        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{obj['class']} {obj['confidence']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for person in persons:
            is_holding = False
            px1, py1, px2, py2 = person['box']

            for obj in objects:
                if is_near(person['box'], obj['box']):
                    is_holding = True
                    ox1, oy1, ox2, oy2 = obj['box']
                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                    cv2.putText(frame, obj['class'], (ox1, oy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ✅ Draw the person box after checking
            if is_holding:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(frame, "Person is holding something", (px1, py1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Processed Frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved annotated video to: {output_path}")


# Example usage:
process_image("sample.jpg")
#process_video("input_video.mp4")

