# Twitter Spam Detection Mini-Project

This project demonstrates a simple pipeline to detect spam tweets using Python.
This mini-project was developed as part of my coursework exploring feature extraction and basic machine learning techniques for spam detection.
Note: The dataset included in this repository is intentionally small and synthetic for demonstration purposes. The goal of this mini-project is to illustrate the machine learning pipeline rather than build a production spam detection system.

- Loads a tiny fake dataset of tweets
- Extracts features using TF-IDF
- Trains a Random Forest classifier
- Outputs evaluation metrics

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python feature_extract.py`

Dependencies: pandas, scikit-learn, numpy


## Project Structure

twitter-spam-detection/
│
├── feature_extract.py   # Main script for training and evaluation
├── sample_data.csv      # Small demo dataset
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
