# website.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
from small_dataset_model import DeepRecommenderSystem, recommend_movies
from sklearn.preprocessing import LabelEncoder
import json
import plotly
import plotly.express as px
import numpy as np

def load_model_and_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratings_df = pd.read_csv("databases/ml-latest-small/ratings.csv")
    movies_df = pd.read_csv("databases/ml-latest-small/movies.csv")
    
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    ratings_df['userId'] = user_encoder.fit_transform(ratings_df['userId'])
    ratings_df['movieId'] = movie_encoder.fit_transform(ratings_df['movieId'])
    
    model = DeepRecommenderSystem(
        num_users=len(user_encoder.classes_),
        num_movies=len(movie_encoder.classes_)
    ).to(device)
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model.eval()
    
    return model, ratings_df, movies_df, user_encoder, movie_encoder, device
def parse_training_history(filename):
    train_losses = []
    val_losses = []
    epochs = []
    current_epoch = 0
    
    with open(filename, 'r') as f:
        for line in f:
            if 'Train Loss:' in line and 'Val Loss:' in line:
                # Extract both train and val loss
                parts = line.strip().split(' - ')
                train_loss = float(parts[1].split(': ')[1])
                val_loss = float(parts[2].split(': ')[1])
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                epochs.append(current_epoch)
                current_epoch += 1
    
    return epochs, train_losses, val_losses

def create_app():
    app = Flask(__name__)
    
    # Load model and data
    model, ratings_df, movies_df, user_encoder, movie_encoder, device = load_model_and_data()
    
    @app.route('/')
    def home():
        metrics = {}
        with open('small_data_recommendation.txt', 'r') as f:
            for line in f:
                if any(metric in line for metric in ['RMSE:', 'MAE:', 'Precision@10:', 'Recall@10:', 'F-measure:', 'NDCG@10:']):
                    key, value = line.strip().split(': ')
                    key = key.replace('@10', '\n@10')
                    key = key.replace('measure', 'measure\n')
                    metrics[key] = float(value)
        return render_template('home.html', metrics=metrics)

    @app.route('/recommend', methods=['GET', 'POST'])
    def recommend():
        if request.method == 'POST':
            user_id = int(request.form['user_id'])
            encoded_user_id = user_encoder.transform([user_id])[0]
            movie_ids = ratings_df['movieId'].unique()
            
            recommendations = recommend_movies(
                model, encoded_user_id, movie_ids, 
                movies_df, device, movie_encoder
            )
            return render_template('recommend.html', recommendations=recommendations, user_id=user_id)
        return render_template('recommend.html')

    @app.route('/visualize')
    def visualize():
        # 1. Training Progress
        epochs, train_losses, val_losses = parse_training_history('small_data_recommendation.txt')
        fig1 = px.line(
            {
                'Epoch': epochs + epochs,
                'Loss': train_losses + val_losses,
                'Type': ['Training']*len(epochs) + ['Validation']*len(epochs)
            },
            x='Epoch', y='Loss', color='Type',
            title='Model Training Progress'
        )

        # 2. User Activity Distribution
        user_activity = ratings_df['userId'].value_counts()
        fig2 = px.histogram(
            user_activity,
            title='User Activity Distribution',
            labels={'value': 'Number of Ratings', 'count': 'Number of Users'},
            nbins=50
        )

        # 3. Rating Distribution by Genre
        movies_with_ratings = pd.merge(ratings_df, movies_df, on='movieId')
        movies_with_ratings['genre'] = movies_with_ratings['genres'].str.split('|')
        genre_ratings = movies_with_ratings.explode('genre')
        fig3 = px.box(
            genre_ratings,
            x='genre',
            y='rating',
            title='Rating Distribution by Genre'
        )

        # 4. Rating Trends Over Time
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        ratings_over_time = ratings_df.set_index('timestamp')['rating'].resample('M').mean()
        fig4 = px.line(
            ratings_over_time,
            title='Average Rating Trend Over Time',
            labels={'value': 'Average Rating', 'timestamp': 'Date'}
        )

        # 5. Movie Popularity vs Average Rating
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['movieId', 'rating_count', 'avg_rating']
        fig5 = px.scatter(
            movie_stats,
            x='rating_count',
            y='avg_rating',
            title='Movie Popularity vs Average Rating',
            labels={'rating_count': 'Number of Ratings', 'avg_rating': 'Average Rating'},
            opacity=0.6
        )

        # Convert plots to JSON
        plots = [fig1, fig2, fig3, fig4, fig5]
        graphJSON = json.dumps([fig.to_dict() for fig in plots], cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('visualize.html', graphJSON=graphJSON)

    @app.route('/about')
    def about():
        return render_template('about.html')

    # Update CSP headers
    @app.after_request
    def add_security_headers(response):
        csp = {
            'default-src': ['\'self\''],
            'script-src': ['\'self\'', '\'unsafe-inline\'', '\'unsafe-eval\'', 
                          'https://cdn.plot.ly', 'https://cdn.jsdelivr.net'],
            'style-src': ['\'self\'', '\'unsafe-inline\'', 'https://cdn.jsdelivr.net'],
            'img-src': ['\'self\'', 'data:', 'https:', 'blob:'],
            'font-src': ['\'self\'', 'data:', 'https:'],
            'connect-src': ['\'self\'', 'https://cdn.plot.ly']
        }
        
        response.headers['Content-Security-Policy'] = '; '.join(
            f"{key} {' '.join(values)}" for key, values in csp.items()
        )
        return response

    return app  # Added return statement

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)