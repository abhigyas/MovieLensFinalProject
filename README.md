# Movie Recommendation System

A deep learning-based movie recommendation system implemented in PyTorch that provides personalized movie recommendations with explanations.

## Features

- Deep neural network architecture with user and movie embeddings
- Explainable recommendations with attention mechanisms
- Interactive web interface for recommendations
- Comprehensive evaluation metrics (RMSE, MAE, Precision@10, Recall@10, F-measure, NDCG@10)
- Training progress visualization
- Embedding space visualization

## Requirements

- Python 3.8+
- PyTorch
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib 
- seaborn
- plotly

Install dependencies:
```bash
pip install -r requirements.txt
```
## Dataset Setup

- Create a databases directory in the project root:
```bash
mkdir -p databases/ml-latest-small
mkdir -p databases/ml-latest
```
- Download the MovieLens datasets: https://grouplens.org/datasets/movielens/latest/
- We are using the small dataset for this project as this is the only dataset that our devices can run
- Extract the datasets, and put the contents of both the small and large datasets (the csv files) into ml-latest-small and ml-latest respectively

## Project 
.
├── databases/                  # Dataset directory (create this)
    ├── ml-latest-small
        ├── movies.csv 
        ├── ratings.csv
        ├── rating.csv
        ├── README.txt
        ├── tags.csv
    ├── ml-latest
        ├── genome-scores.csv
        ├── genome-tags.csv
        ├── movies.csv 
        ├── ratings.csv
        ├── rating.csv
        ├── README.txt
        ├── tags.csv
├── static/                    # Static web assets
    ├── main.css
├── templates/                 # Flask HTML templates  
    ├── about.html
    ├── base.html
    ├── home.html
    ├── recommend.html
    ├── visualize.html
├── small_dataset_model.py    # Core recommendation model
├── large_dataset_model.py    # Scaled up model version
├── website.py                # Flask web interface
└── README.md                 # This file
├── small_data_recommendation.txt    # output after executing small_dataset_model.py

## Usage

- Train the model using either the small or large dataset:

- For small dataset (Run small dataset for the best results as the findings will be shown on the website)
```bash
python small_dataset_model.py
```
- For large dataset (Note: The large dataset model has not been tested completely due to hardware constraints, so only run with good hardware. This also means that the website will not reflect large dataset findings)
```bash
python large_dataset_model.py
```
- Run the website interface
```bash
python website.py
```
- Open http://localhost:5000 in browser (Terminal should also have a link as output)

## Evaluation Metrics
- The system evaluates recommendations using:
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - Precision@10
    - Recall@10
    - F-measure
    - NDCG@10 (Normalized Discounted Cumulative Gain)

## Web Interface Features

- Get personalized movie recommendations
    - View recommendation explanations
    - Interactive visualizations:
        - Training progress
        - User/Movie embedding spaces
        - Rating distributions
        - Temporal analysis

## Optional Tasks Implemented
- Transparency and Explainability:
    - Attention-based explanation generation
    - Visual analysis of recommendation factors
    - Embedding space visualization

## Important Files

- small_dataset_model.py: Core model for MovieLens small dataset
- large_dataset_model.py: Scaled version for full MovieLens dataset
- website.py: Flask web application
- templates: HTML templates for web interface
- static: CSS, JavaScript and images
- databases: Dataset storage (not included in repo)

## Contributors

- Abhigya Sinha
- Bhavya Patel
- Paul Kanyuch