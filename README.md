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
- seaborn
- plotly

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project 
- paper
    - recommender_systems_paper.tex    # LaTex file of our research paper
- data/                                
    - movies.csv 
    - ratings.csv
- static/                    # Static web assets
    - main.css
- templates/                 # Flask HTML templates  
    - about.html
    - base.html
    - home.html
    - recommend.html
    - visualize.html
- small_dataset_model.py    # Core recommendation model
- website.py                # Flask web interface
- README.md                 # This file
- small_data_recommendation.txt    # output after executing small_dataset_model.py

## Usage

- Train the model using the dataset:

- For dataset 
```bash
python small_dataset_model.py
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
- website.py: Flask web application
- templates: HTML templates for web interface
- static: CSS, JavaScript and images
- databases: Dataset storage

## Contributors

- Abhigya Sinha
- Bhavya Patel
- Paul Kanyuch

## This project has been deployed!
You can access the deployed project at [MovieLens Recommender System](https://movielens-recommender-system-52253e175e9b.herokuapp.com/recommend)