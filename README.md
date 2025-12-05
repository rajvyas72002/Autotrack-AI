# Autotrack-AI
AI system for real-time vehicle tracking, speed measurement, and number plate recognition with automated logging and analytics.

# 🚗 AutoTrack AI

**Real-time vehicle detection, tracking, speed estimation, and number plate extraction using YOLOv5, Deep SORT, EasyOCR, and OpenCV.**

---

## 🔥 Features
- YOLOv5 vehicle detection  
- Deep SORT object tracking with unique IDs  
- Number plate extraction using EasyOCR  
- Speed estimation using movement + timestamps  
- Entry/exit time logging  
- Cropped vehicle image saving  
- Automatic CSV report generation  
- Optional LLM-based summaries (Ollama)

---

## 🛠 Tech Stack
- Python  
- YOLOv5 (PyTorch)  
- Deep SORT  
- OpenCV  
- EasyOCR  
- Pandas  
- CUDA (optional)

---

## 📂 Project Structure

AutoTrack-AI/
│── yolov5/
│── deep_sort/
│── data/
│── output/
│ ├── crops/
│ ├── report.csv
│── main.py
│── utils.py
│── requirements.txt
│── README.md




---

## ▶️ How to Run
```bash
git clone https://github.com/yourusername/AutoTrack-AI
cd AutoTrack-AI
pip install -r requirements.txt
python main.py
