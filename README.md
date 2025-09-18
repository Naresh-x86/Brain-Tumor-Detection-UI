# Brain Tumor Detection App

A user-friendly desktop application for brain tumor detection using MRI images, powered by a Convolutional Neural Network (CNN) in TensorFlow. This repository includes:
- A ready-to-use GUI (`app.py`) for image analysis
- Pre-trained model weights (`brain_tumor_detection_model.h5`)
- Jupyter notebook for model training (`model.ipynb`)
- Dataset and training code links

---

## 🚀 Getting Started

### 1. Clone the Repository
```sh
git clone git@github-naresh-x86:Naresh-x86/Brain-Tumor-Detection-UI.git
cd Brain-Tumor-Detection-UI
```

### 2. Create and Activate a Virtual Environment (Recommended)
```sh
python -m venv venv
# On Windows PowerShell
.\venv\Scripts\Activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

---

## 🖼️ Run the App

### Option 1: Directly from Python
```sh
python app.py
```

### Option 2: Compile to Standalone EXE (Windows)
1. Install PyInstaller:
    ```sh
    pip install pyinstaller
    ```
2. Compile:
    ```sh
    pyinstaller --onefile app.py
    ```
3. Move the generated `app.exe` from `/dist` to the main folder (it must be in the same folder as `brain_tumor_detection_model.h5` for inference).

---

## 🔄 Retrain the Model (Optional)
- Open and run all cells in `model.ipynb` to retrain the CNN model.
- The notebook covers data loading, augmentation, training, evaluation, and saving the model weights.

---

## 📂 Dataset & Training Code
- **Dataset:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Training Notebook Reference:** [Vidhi1290/Brain-Tumor-Detection (GitHub)](https://github.com/Vidhi1290/Brain-Tumor-Detection/tree/main)
- These links are also provided in the `.url` files in the repository for convenience.

---

## ⚠️ Notes & Important Information
- The app requires `brain_tumor_detection_model.h5` in the same directory as `app.py` or `app.exe` for predictions.
- The dataset included despite its large size; If cloning is difficult, download it from Kaggle and place it in the `dataset/` folder as structured in the notebook.
- The GUI uses [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for a modern look. Install it via `pip install customtkinter` if not present.
- For retraining, ensure you have sufficient system resources (RAM/GPU) and the dataset downloaded.
- If using multiple Python environments/accounts, ensure you activate the correct environment before installing dependencies or running the app.
- For SSH setup, use your custom SSH config as shown above for cloning and pushing changes.

---

## 💡 Features
- Upload MRI images and get instant predictions for tumor type (glioma, meningioma, notumor, pituitary)
- Visual confidence bars for each class
- Easy retraining and model updating via Jupyter notebook
- Standalone executable option for Windows users

---