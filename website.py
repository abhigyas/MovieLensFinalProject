# website.py
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
from small_dataset_model import ExplainableRecommenderSystem, explain_recommendation, recommend_movies
from sklearn.preprocessing import LabelEncoder
import json
import plotly
import plotly.express as px
import numpy as np
import gc
from functools import lru_cache

def load_model_and_data():
    device = torch.device("cpu")
    ratings_df = pd.read_csv("data/ratings.csv")
    movies_df = pd.read_csv("data/movies.csv")
    
    # Load encoders
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    ratings_df['userId'] = user_encoder.fit_transform(ratings_df['userId'])
    ratings_df['movieId'] = movie_encoder.fit_transform(ratings_df['movieId'])
    
    # Match original architecture sizes
    model = ExplainableRecommenderSystem(
        num_users=len(user_encoder.classes_),
        num_movies=len(movie_encoder.classes_),
        embedding_size=128  # Match saved model size
    ).to(device)
    
    # Add weights_only=True for security
    model.load_state_dict(
        torch.load(
            'best_model.pth',
            map_location=device,
            weights_only=True
        )
    )
    model.eval()
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, ratings_df, movies_df, user_encoder, movie_encoder, device

# Cache training history parsing
@lru_cache(maxsize=1)
def parse_training_history(filename):
    train_losses = []
    val_losses = []
    epochs = []
    current_epoch = 0
    
    try:
        with open(filename, 'r', encoding='latin-1') as f:
            for line in f:
                if 'Train Loss:' in line and 'Val Loss:' in line:
                    try:
                        parts = line.strip().split(' - ')
                        train_loss = float(parts[1].split(': ')[1])
                        val_loss = float(parts[2].split(': ')[1])
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        epochs.append(current_epoch)
                        current_epoch += 1
                    except (IndexError, ValueError) as e:
                        continue
    except Exception as e:
        return [], [], []
        
    return epochs, train_losses, val_losses

def create_app():
    app = Flask(__name__)
    app.config['PORT'] = int(os.environ.get('PORT', 8000))
    
    # Load model and data
    model, ratings_df, movies_df, user_encoder, movie_encoder, device = load_model_and_data()
    
    @app.route('/')
    def home():
        metrics = {}
        try:
            with open('small_data_recommendation.txt', 'r', encoding='latin-1') as f:
                for line in f:
                    if any(metric in line for metric in ['RMSE:', 'MAE:', 'Precision@10:', 'Recall@10:', 'F-measure:', 'NDCG@10:']):
                        key, value = line.strip().split(': ')
                        key = key.replace('@10', '\n@10')
                        key = key.replace('measure', 'measure\n')
                        metrics[key] = float(value)
        except Exception as e:
            app.logger.error(f"Error reading metrics: {str(e)}")
            metrics = {"Error": "Could not load metrics"}
        
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
            
            # Generate explanations for recommendations
            explanations = []
            for movie in recommendations[:3]:  # Get explanations for top 3
                explanation = explain_recommendation(
                    model,
                    encoded_user_id,
                    movie['movieId'],
                    device
                )
                
                try:
                    # Parse the explanation text to extract values
                    lines = explanation.split('\n')
                    # Find lines containing the relevant information
                    for line in lines:
                        if "User preferences:" in line:
                            user_importance = float(line.split(': ')[1])
                        elif "Movie characteristics:" in line:
                            movie_importance = float(line.split(': ')[1])
                    
                    explanations.append({
                        'movie': movie['title'],
                        'explanation': {
                            'user_importance': user_importance,
                            'movie_importance': movie_importance
                        },
                        'rating': movie['predicted_rating'],
                        'full_explanation': explanation  # Keep full explanation text
                    })
                except Exception as e:
                    print(f"Error parsing explanation: {str(e)}")
                    continue
            
            if explanations:  # Only create visualization if we have explanations
                # Create attention visualization
                attention_fig = px.bar(
                    pd.DataFrame([
                        {'Factor': 'User Preferences', 'Weight': explanations[0]['explanation']['user_importance']},
                        {'Factor': 'Movie Characteristics', 'Weight': explanations[0]['explanation']['movie_importance']}
                    ]),
                    x='Factor',
                    y='Weight',
                    title='Recommendation Factors'
                )
                attention_plot = json.dumps(attention_fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
            else:
                attention_plot = None
            
            return render_template(
                'recommend.html',
                recommendations=recommendations,
                explanations=explanations,
                attention_plot=attention_plot,
                user_id=user_id
            )
        return render_template('recommend.html')

    # Cache visualization computation
    @lru_cache(maxsize=1)
    def generate_visualizations(ratings_df, movies_df, model, device):
        try:
            plots = []
            # Add existing visualization code but with smaller samples
            user_sample = ratings_df['userId'].unique()[:50]  # Reduced from 100
            movie_sample = ratings_df['movieId'].unique()[:50]  # Reduced from 100
            
            with torch.no_grad():
                user_embeddings = model.user_embedding(torch.tensor(user_sample).to(device)).cpu().numpy()
                movie_embeddings = model.movie_embedding(torch.tensor(movie_sample).to(device)).cpu().numpy()
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Return plots
            return plots
        except Exception as e:
            return []

    @app.route('/visualize')
    def visualize():
        try:
            # Generate plots with caching
            plots = generate_visualizations(ratings_df, movies_df, model, device)
            if not plots:
                return render_template('error.html', message="Could not generate visualizations")
                
            graphJSON = json.dumps([fig.to_dict() for fig in plots], cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('visualize.html', graphJSON=graphJSON)
        except Exception as e:
            app.logger.error(f"Error in visualization: {str(e)}")
            return render_template('error.html', message="Error generating visualizations")

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

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)