🧠 Nidan AI – Medical Image Analysis System

🚀 Overview

Nidan AI is an intelligent web-based medical image analysis system that uses Deep Learning to automatically detect, classify, and segment diseases from medical images.

The system supports:

- 👁️ Eye Disease Detection (Diabetic Retinopathy)
- 🧠 Brain Tumor Detection & Segmentation
- 🤖 Automatic Image Type Classification (Eye vs Brain)

Built using Flask + TensorFlow + OpenCV, this project demonstrates real-world AI application in healthcare.

---

🎯 Key Features

🔍 1. Automatic Image Type Detection

- Detects whether uploaded image is:
  - Eye image
  - Brain MRI scan
- Automatically selects the correct model

---

👁️ 2. Eye Disease Detection

- Detects Diabetic Retinopathy
- Provides:
  - Prediction label
  - Confidence score

---

🧠 3. Brain Tumor Detection

- Classifies MRI images into:
  - Tumor
  - No Tumor

---

🧩 4. Tumor Segmentation (Advanced Feature)

- Highlights tumor region using segmentation model
- Displays:
  - Original image
  - Segmented image (side-by-side)
- Includes visual effects (highlight/animation)

---

🌐 5. User-Friendly Web Interface

- Built using Flask + Tailwind CSS
- Features:
  - Image upload
  - Result visualization
  - Download segmented output

---

💾 6. Model Auto-Download Support

- Models are hosted on cloud (Google Drive)
- Automatically downloaded using script

---

🛠️ Tech Stack

Technology | Purpose

Python | Core Programming

Flask | Web Framework

TensorFlow / Keras | Deep Learning Models

OpenCV | Image Processing

NumPy | Numerical Computation

Tailwind CSS | UI Design

---

📥 Download Models

⚠️ Models are not included in this repository due to large size.

📁 Folder Structure Required:

models/
├── my_model.keras
├── best_brain_tumor_model.keras

🔗 Download from Google Drive:

Steps:

Download both models

Brain Tumor (best_brain_tumor_model.keras)

https://drive.google.com/uc?id=1qGwS5KM4NOaLn25hVkeOhVkusNWwlT2D

eye model (my_model.keras)

https://drive.google.com/uc?id=1EpOJ4-B6zeK08nZivm-Wb4zT7lL4VNdk

Place them inside the models/ folder

---

⚙️ Setup Guide

✅ Step 1: Clone Repository

git clone https://github.com/your-username/Nidan-AI.git
cd Nidan-AI

---

✅ Step 2: Create Virtual Environment

python -m venv venv
venv\Scripts\activate

---

✅ Step 3: Install Dependencies

pip install -r requirements.txt

---

✅ Step 4: Download Models

python download_models.py

👉 This will download models into:

models/

---

✅ Step 5: Run Application

python app.py

---

🌐 Access Web App

Open browser and go to:

http://127.0.0.1:5000/

---

🎨 Tailwind CSS Setup

Your project uses Tailwind CSS for UI styling.

Option 1 (Recommended – CDN)

Add this inside <head> of your HTML:

<script src="https://cdn.tailwindcss.com"></script>


👉 No installation required ✅

Option 2 (Advanced – Full Setup)

Install Node.js

Install Tailwind:

npm install -D tailwindcss

npx tailwindcss init

Configure tailwind.config.js

Build CSS:

npx tailwindcss -i ./static/css/input.css -o ./static/css/output.css --watch

---

📸 How It Works

1. Upload a medical image
2. System detects image type
3. Appropriate model is selected
4. Prediction is generated
5. If tumor → segmentation is applied
6. Results displayed on UI

---

⚠️ Important Notes

- Models are not stored in GitHub due to size limitations
- Ensure models are downloaded before running app
- Recommended Python version: 3.10

---

📌 Future Enhancements

- Add more diseases (COVID, TB, Pneumonia)
- Deploy on cloud (AWS / Render)
- Improve model accuracy
- Add real-time camera input
- Integrate with hospital systems

---

🎓 Use Case

- Academic Projects (B.Tech / M.Tech)
- Healthcare AI Research
- Portfolio Project for Placements

---

🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

📧 Contact

Developer: Devansh Sharma
Field: AI / ML / Full Stack Development

---

⭐ Acknowledgment

- TensorFlow & Keras Community
- Open-source medical datasets
- Flask Framework

---

⭐ If you like this project

Give it a ⭐ on GitHub!

---
