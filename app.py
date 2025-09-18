import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import os

# Configure CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Global font configuration
UI_FONT_FAMILY = "Segoe UI"  # Clean Windows font
UI_FONT_SIZE_SMALL = 12
UI_FONT_SIZE_MEDIUM = 14
UI_FONT_SIZE_LARGE = 18
UI_FONT_SIZE_TITLE = 24

class BrainTumorDetectionApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Brain Tumor Detection")
        self.root.geometry("710x570")
        self.root.resizable(False, False)
        
        # Initialize variables
        self.model = None
        self.current_image_path = None
        self.categories = ["glioma", "meningioma", "notumor", "pituitary"]
        self.image_size = (150, 150)
        
        # Load model
        self.load_model()
        
        # Setup UI
        self.setup_ui()
        
    def load_model(self):
        """Load the trained brain tumor detection model"""
        try:
            model_path = "brain_tumor_detection_model.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print("Model loaded successfully!")
            else:
                messagebox.showerror("Error", f"Model file not found: {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(main_frame, text="Brain Tumor Detection", 
                           font=ctk.CTkFont(family=UI_FONT_FAMILY, size=UI_FONT_SIZE_TITLE, weight="bold"))
        title.pack(pady=(20, 10))
        
        # Buttons frame
        buttons_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        buttons_frame.pack(pady=10)
        
        self.upload_btn = ctk.CTkButton(buttons_frame, text="Upload Image", 
                                       command=self.upload_image, width=120, height=30,
                                       font=ctk.CTkFont(family=UI_FONT_FAMILY, size=UI_FONT_SIZE_MEDIUM))
        self.upload_btn.pack(side="left", padx=20, pady=20)
        
        self.predict_btn = ctk.CTkButton(buttons_frame, text="Analyze", 
                                        command=self.predict_tumor, width=120, height=30,
                                        state="disabled",
                                        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=UI_FONT_SIZE_MEDIUM))
        self.predict_btn.pack(side="left", padx=10, pady=20)
        
        self.clear_btn = ctk.CTkButton(buttons_frame, text="Clear", 
                                      command=self.clear_all, width=120, height=30,
                                      font=ctk.CTkFont(family=UI_FONT_FAMILY, size=UI_FONT_SIZE_MEDIUM))
        self.clear_btn.pack(side="left", padx=20, pady=20)
        
        # Content frame
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, pady=20)
        
        # Image display
        self.image_label = ctk.CTkLabel(content_frame, text="No image selected", 
                                       width=300, height=300,
                                       font=ctk.CTkFont(family=UI_FONT_FAMILY, size=UI_FONT_SIZE_MEDIUM))
        self.image_label.pack(side="left", padx=(20, 0), pady=20)
        
        # Results display
        self.results_frame = ctk.CTkFrame(content_frame)
        self.results_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        # Results title
        results_title = ctk.CTkLabel(self.results_frame, text="Results", 
                                   font=ctk.CTkFont(family=UI_FONT_FAMILY, size=UI_FONT_SIZE_LARGE, weight="bold"))
        results_title.pack(pady=(20, 10))
        
        # Prediction labels (will be created dynamically)
        self.prediction_labels = []
        
    def upload_image(self):
        """Handle image upload"""
        file_types = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        file_path = filedialog.askopenfilename(title="Select Brain MRI Image", filetypes=file_types)
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_btn.configure(state="normal")
            
    def display_image(self, image_path):
        """Display the uploaded image"""
        try:
            pil_image = Image.open(image_path)
            pil_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image file")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            return image
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    def predict_tumor(self):
        """Make prediction on the uploaded image"""
        if not self.model or not self.current_image_path:
            messagebox.showerror("Error", "Model or image not available!")
            return
        
        try:
            self.predict_btn.configure(text="Analyzing...", state="disabled")
            self.root.update()
            
            processed_image = self.preprocess_image(self.current_image_path)
            predictions = self.model.predict(processed_image, verbose=0)
            
            self.display_results(predictions[0])
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        finally:
            self.predict_btn.configure(text="Analyze", state="normal")
    
    def display_results(self, predictions):
        """Display prediction results with progress bars"""
        # Clear previous results
        for label in self.prediction_labels:
            label.destroy()
        self.prediction_labels.clear()
        
        # Sort predictions by confidence
        pred_pairs = list(zip(self.categories, predictions))
        pred_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (category, prob) in enumerate(pred_pairs):
            # Category name and percentage
            text = f"{category.title()}: {prob:.1%}"
            label = ctk.CTkLabel(self.results_frame, text=text, 
                               font=ctk.CTkFont(family=UI_FONT_FAMILY, size=UI_FONT_SIZE_MEDIUM))
            label.pack(pady=5, anchor="w", padx=20)
            self.prediction_labels.append(label)
            
            # Progress bar container
            bar_frame = ctk.CTkFrame(self.results_frame, height=8, corner_radius=4)
            bar_frame.pack(fill="x", pady=(0, 10), padx=20)
            bar_frame.pack_propagate(False)
            
            # Progress bar (filled portion)
            bar_width = int(prob * 260)  # 260 is approximate usable width
            color = self.get_probability_color(prob)
            
            if bar_width > 0:
                progress_bar = ctk.CTkLabel(bar_frame, text="", width=bar_width, height=8,
                                          fg_color=color, corner_radius=4)
                progress_bar.pack(side="left", anchor="w")
                self.prediction_labels.append(progress_bar)
            
            self.prediction_labels.append(bar_frame)
    
    def get_probability_color(self, probability):
        """Get color based on probability value"""
        if probability >= 0.7:
            return "#4CAF50"  # Green
        elif probability >= 0.4:
            return "#FF9800"  # Orange
        elif probability >= 0.2:
            return "#FFC107"  # Yellow
        else:
            return "#F44336"  # Red
    
    def clear_all(self):
        """Clear all data and reset the interface"""
        self.current_image_path = None
        self.image_label.configure(image="", text="No image selected")
        self.image_label.image = None
        self.predict_btn.configure(state="disabled")
        
        for label in self.prediction_labels:
            label.destroy()
        self.prediction_labels.clear()
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    app = BrainTumorDetectionApp()
    app.run()

if __name__ == "__main__":
    main()