import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from numba import jit

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

class DeepRecommenderSystem(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=128):
        super(DeepRecommenderSystem, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        
        # Deep Neural Network layers
        self.layers = nn.Sequential(
            nn.Linear(2 * embedding_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, users, movies):
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)
        concatenated = torch.cat([user_embedded, movie_embedded], dim=1)
        return self.layers(concatenated).squeeze()

def get_dataloaders(train_dataset, val_dataset, batch_size=256):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = GradScaler()  # For mixed precision training
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            users = batch["users"].to(device, non_blocking=True)
            movies = batch["movies"].to(device, non_blocking=True)
            ratings = batch["ratings"].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Use mixed precision training
            with autocast():
                predictions = model(users, movies)
                loss = criterion(predictions * 5.0, ratings)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

@jit(nopython=True)
def calculate_ndcg_fast(true_ratings, predicted_ratings, k=10):
    # Optimized NDCG calculation
    indices = np.argsort(predicted_ratings)[-k:][::-1]
    dcg = 0
    for i, idx in enumerate(indices):
        dcg += (2**true_ratings[idx] - 1) / np.log2(i + 2)
    
    ideal = np.sort(true_ratings)[::-1]
    idcg = 0
    for i in range(min(k, len(ideal))):
        idcg += (2**ideal[i] - 1) / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0

def calculate_ndcg(true_ratings, predicted_ratings, k=10):
    """
    Calculate NDCG@k for a list of predictions
    
    Args:
        true_ratings: List of true ratings
        predicted_ratings: List of predicted ratings
        k: Number of items to consider
    """
    # Sort predictions and get top k indices
    top_k_indices = np.argsort(predicted_ratings)[-k:][::-1]
    
    # Get DCG
    dcg = 0
    for i, idx in enumerate(top_k_indices):
        rel = true_ratings[idx]
        dcg += (2**rel - 1) / np.log2(i + 2)  # i+2 because i starts from 0
    
    # Get IDCG (sort true ratings in descending order)
    ideal_order = np.argsort(true_ratings)[-k:][::-1]
    idcg = 0
    for i, idx in enumerate(ideal_order):
        rel = true_ratings[idx]
        idcg += (2**rel - 1) / np.log2(i + 2)
    
    # Calculate NDCG
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

def calculate_metrics(model, val_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    # Store user-specific predictions and actual items
    user_test_items = defaultdict(set)
    user_recommendations = defaultdict(list)
    
    with torch.no_grad():
        for batch in val_loader:
            users = batch["users"].to(device)
            movies = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)
            
            output = model(users, movies)
            scaled_output = output * 5.0  # Scale back to 0-5 range
            
            predictions.extend(scaled_output.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())
            
            # Store both predicted and true ratings
            for user, movie, pred, true in zip(users.cpu().numpy(), 
                                             movies.cpu().numpy(),
                                             scaled_output.cpu().numpy(),
                                             ratings.cpu().numpy()):
                user_test_items[user].add(movie)
                user_recommendations[user].append((movie, pred, true))
    
    # Calculate RMSE
    rmse = root_mean_squared_error(actuals, predictions)
    
    # Calculate Precision and Recall for each user
    precisions = []
    recalls = []
    
    for user_id in user_test_items.keys():
        # Sort recommendations by predicted rating
        user_recs = user_recommendations[user_id]
        user_recs.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item[0] for item in user_recs]  # Get just the movie IDs
        
        # Calculate precision and recall
        p, r = calculate_precision_recall(
            user_id=user_id,
            test_items=user_test_items[user_id],
            recommended_items=recommended_items,
            k=10
        )
        precisions.append(p)
        recalls.append(r)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    # Calculate F-measure
    f_measure = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    ndcg_scores = []
    for user_id in user_test_items.keys():
        user_recs = user_recommendations[user_id]
        # Sort by predicted ratings
        user_recs.sort(key=lambda x: x[1], reverse=True)
        
        # Get aligned predicted and true ratings
        pred_ratings = np.array([pred for _, pred, _ in user_recs[:10]])
        true_ratings = np.array([true for _, _, true in user_recs[:10]])
        
        ndcg = calculate_ndcg(true_ratings, pred_ratings, k=10)
        ndcg_scores.append(ndcg)
    
    avg_ndcg = np.mean(ndcg_scores)
    
    return rmse, avg_precision, avg_recall, f_measure, avg_ndcg

def calculate_precision_recall(user_id, test_items, recommended_items, k=10):
    # Args:
    #     user_id: The user ID
    #     test_items: Set of items in test set for this user
    #     recommended_items: List of recommended items (top-10)
    #     k: Number of recommendations to consider (10)

    # Take only first k recommendations
    recommended_k = recommended_items[:k]
    
    # Count how many recommended items are in test set
    hits = len(set(recommended_k) & set(test_items))
    
    # Precision = hits / k
    precision = hits / k if k > 0 else 0
    
    # Recall = hits / total test items
    recall = hits / len(test_items) if test_items else 0
    
    return precision, recall

def recommend_movies(model, user_id, movie_ids, df_movies, device, movie_encoder, top_k=10):
    # Recommend movies for a user
    # Args:
    #     model: Trained model
    #     user_id: User ID (encoded)
    #     movie_ids: List of encoded movie IDs
    #     df_movies: Original movies dataframe
    #     device: torch device
    #     movie_encoder: LabelEncoder used for movie IDs
    #     top_k: Number of recommendations to return

    model.eval()
    
    # Create tensors for prediction
    user_tensor = torch.tensor([user_id] * len(movie_ids), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(movie_ids, dtype=torch.long).to(device)
    
    with torch.no_grad():
        predictions = model(user_tensor, movie_tensor)
        predictions = predictions.cpu().numpy() * 5.0  # Scale to 0-5 range
    
    # Create movie recommendations
    movie_preds = list(zip(movie_ids, predictions))
    movie_preds.sort(key=lambda x: x[1], reverse=True)
    top_movies = movie_preds[:top_k]
    
    # Get movie details
    recommended_movies = []
    for encoded_movie_id, pred_rating in top_movies:
        # Convert encoded ID back to original movie ID
        original_movie_id = movie_encoder.inverse_transform([encoded_movie_id])[0]
        movie_info = df_movies[df_movies['movieId'] == original_movie_id]
        if not movie_info.empty:
            movie_info = movie_info.iloc[0]
            recommended_movies.append({
                'title': movie_info['title'],
                'predicted_rating': pred_rating
            })
    
    return recommended_movies

def main():
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading data...")
    ratings_df = pd.read_csv("databases/ml-latest-small/ratings.csv")
    movies_df = pd.read_csv("databases/ml-latest-small/movies.csv")
    
    # Encode user and movie IDs
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    ratings_df['userId'] = user_encoder.fit_transform(ratings_df['userId'])
    ratings_df['movieId'] = movie_encoder.fit_transform(ratings_df['movieId'])
    
    # Split data
    train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = MovieLensDataset(
        users=train_df.userId.values,
        movies=train_df.movieId.values,
        ratings=train_df.rating.values
    )
    
    val_dataset = MovieLensDataset(
        users=val_df.userId.values,
        movies=val_df.movieId.values,
        ratings=val_df.rating.values
    )
    
    # Create dataloaders
    batch_size = 512 if torch.cuda.is_available() else 64
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size)
    
    # Initialize model
    model = DeepRecommenderSystem(
        num_users=len(user_encoder.classes_),
        num_movies=len(movie_encoder.classes_)
    ).to(device)
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    
    # Calculate metrics
    print("\nCalculating metrics...")
    rmse, precision, recall, f_measure, ndcg = calculate_metrics(model, val_loader, device)
    print(f"RMSE: {rmse:.4f}")
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")
    print(f"F-measure: {f_measure:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")
    
    print("\nGenerating sample recommendations...")
    sample_user_id = 1
    movie_ids = ratings_df['movieId'].unique()
    
    # Get recommendations once
    recommendations = recommend_movies(model, sample_user_id, movie_ids, movies_df, device, movie_encoder)
    
    # Print recommendations once in a clear format
    print(f"\nTop 10 recommended movies for user {sample_user_id}:")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie['title']} - Predicted rating: {movie['predicted_rating']:.2f}")

if __name__ == "__main__":
    main()