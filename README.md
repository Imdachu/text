# Soil Analysis Project

This project analyzes various aspects of soil health using computer vision techniques. It classifies soil texture, estimates moisture levels, detects nutrient deficiencies, and more.

## Project Structure

soil_analysis_project/
│
├── RandomForestClassifier.py # Script to train the soil texture classifier
├── scaler.pkl # Scaler for feature normalization
├── soil_texture_model.pkl # Trained RandomForestClassifier model
├── soil_analysis.py # Main script to analyze soil images
└── README.md # Project instructions and documentation



## Requirements

- Python 3.x
- OpenCV
- NumPy
- scikit-learn
- joblib
- matplotlib

You can install the required packages using pip:


pip install opencv-python numpy scikit-learn joblib matplotlib
Training the Soil Texture Classifier
To train the soil texture classifier, you need a dataset with labeled images of different soil types. Ensure your dataset is structured as follows:

dataset/
    Black_Soil/
        image1.jpg
        image2.jpg
        ...
    Cinder_Soil/
        image1.jpg
        image2.jpg
        ...
    ...


Update data_dir in RandomForestClassifier.py and run:
python RandomForestClassifier.py


This version keeps the essential information while being more concise.


