from ultralytics import YOLO
import cv2


model = YOLO('yolov8x.pt')
TARGET_CLASSES = ['person', 'bottle', 'cup', 'vase', 'wine glass', 'bowl']

# Function to check if bottle/cup is close to a person
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

def process_image(image_path, save_path="output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Image not found.")
        return

    # Check image dimensions
    print(f"Image shape: {img.shape}")

    results = model.predict(source=img, conf=0.25)
    boxes = []

    # Debug: Check the number of detections and the classes
    print(f"Detected {len(results[0].boxes)} objects.")

    for box in results[0].boxes:
        cls_id = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        label = model.names[cls_id]

        # Debug: Print detected class and confidence
        print(f"Detected {label} with confidence {conf}")

        if label in TARGET_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            boxes.append({'class': label, 'box': [x1, y1, x2, y2], 'confidence': conf})

    persons = [obj for obj in boxes if obj['class'] == 'person']
    objects = [obj for obj in boxes if obj['class'] in ['bottle', 'cup', 'vase', 'wine glass', 'bowl']]

    # Add a simple check to confirm rectangles are drawn correctly
    for obj in boxes:
        x1, y1, x2, y2 = obj['box']
        # Just for debugging: draw a rectangle around each detected object
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
                # Draw bounding box around the object (cup/bottle)
                #cv2.rectangle(img, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                #cv2.putText(img, obj['class'], (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw bounding box around the person
            if is_holding:
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(img, "Person is holding something", (px1, py1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
            else:
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(img, "Person", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Debug: Print the number of objects that were drawn with bounding boxes
        print(f"Total objects detected and annotated: {len(boxes)}")

        cv2.imwrite(save_path, img)
        print(f"[INFO] Saved output image: {save_path}")
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_frame(frame):
    # YOLO object detection here
    results = model.predict(source=frame, conf=0.25)
    return frame, results

def process_video(video_path, output_path='output_video.avi'):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Unable to open video file.")
        return

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a video writer to save the output
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Frame skipping: process every 3rd frame to reduce workload
    frame_skip = 3
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip this frame

        # Resize frame to speed up processing
        frame = cv2.resize(frame, (640, 480))

        # Process the frame using YOLO
        results = model.predict(source=frame, conf=0.25)
        boxes = []

        # Debugging: Print frame info
        print(f"Processing frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")

        for box in results[0].boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            label = model.names[cls_id]

            # Debugging: Print detection info
            print(f"Detected {label} with confidence {conf}")

            if label in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                boxes.append({'class': label, 'box': [x1, y1, x2, y2], 'confidence': conf})

        persons = [obj for obj in boxes if obj['class'] == 'person']
        objects = [obj for obj in boxes if obj['class'] in ['bottle', 'cup', 'vase', 'wine glass', 'bowl']]

        # Add a simple check to confirm rectangles are drawn correctly
        for obj in boxes:
            x1, y1, x2, y2 = obj['box']
            # Just for debugging: draw a rectangle around each detected object
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
                    cv2.putText(frame, obj['class'], (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if is_holding:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(frame, "Person is holding something", (px1, py1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame with annotations (Optional for debugging)
        cv2.imshow("Processed Frame", frame)

        # Wait for key press to proceed with the next frame (Esc to exit)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved annotated video to: {output_path}")

# Example usage:

process_image(r"C:\Users\Omar\Desktop\sample.jpg")
process_video(r"C:\Users\Omar\Desktop\input_video.mp4")
