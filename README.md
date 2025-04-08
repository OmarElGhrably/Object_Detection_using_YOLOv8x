# Object_Detection_using_YOLOv8x

This project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and OpenCV to detect when a **person is holding a cup, bottle, or similar object** in an image or video.

It’s designed to work even if the object is close to the person’s body or partially hidden.

---

## 🚀 Features

- Detects objects in images and videos using YOLOv8
- Highlights people who are **holding cups, bottles**
- Saves annotated images/videos with bounding boxes and labels
- Works with both `.jpg` images and `.mp4` videos

---

## 🛠️ Requirements

Make sure you have Python 3.8 or later. Then install the needed packages:

```bash
pip install ultralytics opencv-python

How to Run on an Image
Just rename the image "sample.jpg" and save it in the same directory with the main.py file "process_image("sample.jpg")"

How to Run on an Video
Just rename the video "input_video.mp4" and save it in the same directory with the main.py file "process_video("input_video.mp4")"


Made with ❤️ by Omar Ahmed
