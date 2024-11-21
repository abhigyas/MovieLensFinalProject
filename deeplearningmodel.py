import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error, precision_score, recall_score, ndcg_score
import torch.nn as nn
import sys

# [Previous Dataset and Model classes remain the same...]
class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        return {
            "users": torch.tensor(self.users[item], dtype=torch.long),
            "movies": torch.tensor(self.movies[item], dtype=torch.long),
            "ratings": torch.tensor(self.ratings[item], dtype=torch.float),
        }

class DeepRecommenderModel(nn.Module):
    def __init__(
        self, 
        num_users, 
        num_movies,
        embedding_dim=64,
        layers=[128, 64, 32]
    ):
        super(DeepRecommenderModel, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        self.mlp_layers = []
        input_dim = embedding_dim * 2
        
        for layer_dim in layers:
            self.mlp_layers.extend([
                nn.Linear(input_dim, layer_dim),
                nn.ReLU(),
                nn.BatchNorm1d(layer_dim),
                nn.Dropout(0.2)
            ])
            input_dim = layer_dim
            
        self.mlp = nn.Sequential(*self.mlp_layers)
        
        self.predictor = nn.Sequential(
            nn.Linear(layers[-1], 1),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, users, movies):
        user_embedding = self.user_embedding(users)
        movie_embedding = self.movie_embedding(movies)
        concat = torch.cat([user_embedding, movie_embedding], dim=1)
        mlp_output = self.mlp(concat)
        prediction = self.predictor(mlp_output)
        return prediction * 5.0

# New Evaluation Metrics Class
class RecommenderMetrics:
    def __init__(self, model, device, le_movie, rating_threshold=3.5):
        self.model = model
        self.device = device
        self.le_movie = le_movie
        self.rating_threshold = rating_threshold

    def calculate_rmse(self, data_loader):
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in data_loader:
                users = batch["users"].to(self.device)
                movies = batch["movies"].to(self.device)
                ratings = batch["ratings"].to(self.device)
                
                pred = self.model(users, movies).squeeze()
                predictions.extend(pred.cpu().numpy())
                actuals.extend(ratings.cpu().numpy())
        
        return root_mean_squared_error(actuals, predictions)

    def get_user_recommendations(self, user_id, n=10):
        self.model.eval()
        with torch.no_grad():
            all_movies = torch.arange(len(self.le_movie.classes_)).to(self.device)
            user_tensor = torch.full_like(all_movies, user_id)
            
            predictions = self.model(user_tensor, all_movies).squeeze()
            top_n_indices = torch.topk(predictions, n).indices.cpu().numpy()
            
            return set(self.le_movie.inverse_transform(top_n_indices))

    def calculate_precision_recall_ndcg(self, test_df, k=10):
        precisions = []
        recalls = []
        ndcgs = []
        
        for user_id in test_df['userId'].unique():
            # Get recommendations
            rec_movies = self.get_user_recommendations(user_id, n=k)
            
            # Get actual liked movies from test set
            actual_movies = set(
                test_df[
                    (test_df['userId'] == user_id) & 
                    (test_df['rating'] >= self.rating_threshold)
                ]['movieId'].values
            )
            
            if len(actual_movies) > 0:
                # Calculate precision and recall
                relevant_recs = len(rec_movies.intersection(actual_movies))
                precision = relevant_recs / len(rec_movies)
                recall = relevant_recs / len(actual_movies)
                
                # Calculate NDCG
                recommended_relevance = np.array(
                    [1 if movie in actual_movies else 0 for movie in rec_movies]
                )
                ideal_relevance = np.ones(min(k, len(actual_movies)))
                
                dcg = np.sum(
                    recommended_relevance / np.log2(np.arange(2, len(recommended_relevance) + 2))
                )
                idcg = np.sum(
                    ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2))
                )
                
                ndcg = dcg / idcg if idcg > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                ndcgs.append(ndcg)
        
        return {
            'precision@k': np.mean(precisions),
            'recall@k': np.mean(recalls),
            'ndcg@k': np.mean(ndcgs)
        }

def train_model(model, train_loader, val_loader, epochs=10, device="cuda"):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    metrics = RecommenderMetrics(model, device, le_movie)
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            users = batch["users"].to(device)
            movies = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)
            
            optimizer.zero_grad()
            predictions = model(users, movies).squeeze()
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                users = batch["users"].to(device)
                movies = batch["movies"].to(device)
                ratings = batch["ratings"].to(device)
                
                predictions = model(users, movies).squeeze()
                val_loss += criterion(predictions, ratings).item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Calculate RMSE
        rmse = metrics.calculate_rmse(val_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    df = pd.read_csv("databases/ml-latest-small/ratings.csv")
    
    # Split data
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    
    # Encode users and movies
    global le_movie  # Make le_movie accessible to metrics class
    le_user = LabelEncoder()
    le_movie = LabelEncoder()
    
    # Combine and fit
    combined_users = pd.concat([df_train['userId'], df_val['userId']])
    combined_movies = pd.concat([df_train['movieId'], df_val['movieId']])
    
    le_user.fit(combined_users)
    le_movie.fit(combined_movies)
    
    # Transform data
    df_train['userId'] = le_user.transform(df_train['userId'])
    df_train['movieId'] = le_movie.transform(df_train['movieId'])
    df_val['userId'] = le_user.transform(df_val['userId'])
    df_val['movieId'] = le_movie.transform(df_val['movieId'])
    
    # Create datasets
    train_dataset = MovieLensDataset(
        users=df_train['userId'].values,
        movies=df_train['movieId'].values,
        ratings=df_train['rating'].values
    )
    
    val_dataset = MovieLensDataset(
        users=df_val['userId'].values,
        movies=df_val['movieId'].values,
        ratings=df_val['rating'].values
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = DeepRecommenderModel(
        num_users=len(le_user.classes_),
        num_movies=len(le_movie.classes_),
        embedding_dim=64,
        layers=[128, 64, 32]
    ).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, epochs=10, device=device)
    
    # Load best model and create metrics calculator
    model.load_state_dict(torch.load('best_model.pth'))
    metrics = RecommenderMetrics(model, device, le_movie)
    
    # Calculate final metrics
    print("\nFinal Evaluation Metrics:")
    print(f"RMSE: {metrics.calculate_rmse(val_loader):.4f}")
    
    ranking_metrics = metrics.calculate_precision_recall_ndcg(df_val, k=10)
    print(f"Precision@10: {ranking_metrics['precision@k']:.4f}")
    print(f"Recall@10: {ranking_metrics['recall@k']:.4f}")
    print(f"NDCG@10: {ranking_metrics['ndcg@k']:.4f}")
    
    # Example recommendations
    movies_df = pd.read_csv("databases/ml-latest-small/movies.csv")
    sample_user_id = df_train['userId'].iloc[0]
    print(f"\nRecommendations for user {le_user.inverse_transform([sample_user_id])[0]}:")
    
    recommended_movies = metrics.get_user_recommendations(sample_user_id, n=10)
    for movie_id in recommended_movies:
        movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        print(f"{movie_info['title']} ({movie_info['genres']})")

if __name__ == "__main__":
    main()