from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')

# We're only interested in these classes
TARGET_CLASSES = ['person', 'bottle', 'cup', 'vase', 'wine glass', 'bowl']

# Function to check if two boxes are close to each other
def is_near(person_box, object_box, overlap_threshold=0.05, center_threshold=0.25):
    # Unpack box coordinates
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = object_box

    # Compute areas
    person_area = (px2 - px1) * (py2 - py1)
    object_area = (ox2 - ox1) * (oy2 - oy1)

    # Compute intersection
    inter_x1 = max(px1, ox1)
    inter_y1 = max(py1, oy1)
    inter_x2 = min(px2, ox2)
    inter_y2 = min(py2, oy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Overlap check
    if inter_area / object_area > overlap_threshold:
        return True

    # Center distance check
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2

    dx = abs(pcx - ocx) / (px2 - px1)
    dy = abs(pcy - ocy) / (py2 - py1)

    if dx < center_threshold and dy < center_threshold:
        return True

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
    objects = [obj for obj in boxes if obj['class'] != 'person']

    # Track which objects are being held
    held_object_ids = set()

    # First, figure out which person is holding something
    for i, person in enumerate(persons):
        person['holding'] = False
        px1, py1, px2, py2 = person['box']

        for j, obj in enumerate(objects):
            if is_near(person['box'], obj['box']):
                person['holding'] = True
                held_object_ids.add(j)  # mark this object as being held

    # Now draw all objects — red if held, blue otherwise
    for j, obj in enumerate(objects):
        x1, y1, x2, y2 = obj['box']
        is_held = j in held_object_ids
        color = (0, 0, 255) if is_held else (255, 0, 0)
        label = f"{obj['class']} {'(held)' if is_held else ''}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Now draw all persons — red if holding, green otherwise
    for person in persons:
        x1, y1, x2, y2 = person['box']
        color = (0, 0, 255) if person['holding'] else (0, 255, 0)
        label = "Person holding" if person['holding'] else "Person"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.25)[0]
        persons = []
        objects = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == 'person':
                persons.append({'box': (x1, y1, x2, y2), 'holding': False})
            elif label in ['cup', 'bottle']:
                objects.append({'box': (x1, y1, x2, y2), 'label': label, 'held': False})

        # Check interactions
        for person in persons:
            for obj in objects:
                if is_near(person['box'], obj['box']):
                    person['holding'] = True
                    obj['held'] = True

        # Draw objects
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            color = (0, 0, 255) if obj['held'] else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, obj['label'], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw persons
        for person in persons:
            x1, y1, x2, y2 = person['box']
            color = (0, 0, 255) if person['holding'] else (0, 255, 0)
            label = "Person Holding" if person['holding'] else "Person"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved annotated video to: {output_path}")

# Example usage:
#process_image("sample.jpg")
process_video("input_video.mp4")

