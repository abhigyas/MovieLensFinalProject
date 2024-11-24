import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn import model_selection
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import sys

class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings, all_movie_ids, negative_samples=4):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        self.all_movie_ids = all_movie_ids
        self.negative_samples = negative_samples
        
        # Create user-movie interaction dictionary
        self.user_movies = {}
        for u, m in zip(users, movies):
            if u not in self.user_movies:
                self.user_movies[u] = set()
            self.user_movies[u].add(m)
    
    def __len__(self):
        return len(self.users) * (self.negative_samples + 1)
    
    def __getitem__(self, idx):
        # Get positive sample
        true_idx = idx // (self.negative_samples + 1)
        user = self.users[true_idx]
        
        if idx % (self.negative_samples + 1) == 0:
            # Return positive sample
            movie = self.movies[true_idx]
            rating = self.ratings[true_idx]
        else:
            # Generate negative sample
            while True:
                movie = np.random.choice(self.all_movie_ids)
                if movie not in self.user_movies.get(user, set()):
                    break
            rating = 0.0
        
        return {
            "users": torch.tensor(user, dtype=torch.long),
            "movies": torch.tensor(movie, dtype=torch.long),
            "ratings": torch.tensor(rating, dtype=torch.float),
        }

class ImprovedRecommendationModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_size=256,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2,
    ):
        super(ImprovedRecommendationModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        
        # Popularity bias term
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.user_bias = nn.Embedding(num_users, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Batch normalization for embeddings
        self.bn_user = nn.BatchNorm1d(embedding_size)
        self.bn_movie = nn.BatchNorm1d(embedding_size)

        # Build MLP layers with residual connections
        layers = []
        input_dim = 2 * embedding_size
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dims[-1], 1)
        self.residual = nn.Linear(2 * embedding_size, 1)

    def forward(self, users, movies):
        # Get embeddings
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)
        
        # Get bias terms
        movie_bias = self.movie_bias(movies)
        user_bias = self.user_bias(users)
        
        # Apply batch normalization
        user_embedded = self.bn_user(user_embedded)
        movie_embedded = self.bn_movie(movie_embedded)

        # Concatenate embeddings
        combined = torch.cat([user_embedded, movie_embedded], dim=1)
        
        # Main network path
        x = self.mlp(combined)
        main_output = self.final(x)
        
        # Residual connection
        res_output = self.residual(combined)
        
        # Combine main and residual paths with bias terms
        output = main_output + res_output + movie_bias + user_bias + self.global_bias
        
        return output.squeeze()

def get_recommendations(model, user_id, all_movies, n=10, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Prepare input tensors
        user_tensor = torch.tensor([user_id] * len(all_movies)).to(device)
        movie_tensor = torch.tensor(all_movies).to(device)
        
        # Get predictions
        predictions = model(user_tensor, movie_tensor)
        
        # Get top N recommendations
        top_n_indices = torch.topk(predictions, n).indices.cpu().numpy()
        
        return [all_movies[i] for i in top_n_indices]

def calculate_metrics(model, val_dataset, all_movies, device, k=10):
    precisions = []
    recalls = []
    
    # Get unique users from validation set
    unique_users = np.unique(val_dataset.users)
    
    for user in unique_users:
        # Get actual movies rated by user in validation set
        actual_movies = set(val_dataset.user_movies.get(user, set()))
        if not actual_movies:
            continue
            
        # Get recommended movies
        recommended_movies = set(get_recommendations(model, user, all_movies, n=k, device=device))
        
        # Calculate precision and recall
        if recommended_movies:
            precision = len(actual_movies & recommended_movies) / len(recommended_movies)
            recall = len(actual_movies & recommended_movies) / len(actual_movies)
            
            precisions.append(precision)
            recalls.append(recall)
    
    return np.mean(precisions), np.mean(recalls)

def main():
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv("databases/ml-latest-small/ratings.csv")
    movies_df = pd.read_csv("databases/ml-latest-small/movies.csv")
    
    # Split data
    df_train, df_val = model_selection.train_test_split(
        df, test_size=0.1, random_state=3, stratify=df.rating.values
    )
    
    # Encode users and movies
    from sklearn.preprocessing import LabelEncoder
    le_user = LabelEncoder()
    le_movie = LabelEncoder()
    
    combined_user_ids = pd.concat([df_train['userId'], df_val['userId']])
    combined_movie_ids = pd.concat([df_train['movieId'], df_val['movieId']])
    
    le_user.fit(combined_user_ids)
    le_movie.fit(combined_movie_ids)
    
    df_train['userId'] = le_user.transform(df_train['userId'])
    df_train['movieId'] = le_movie.transform(df_train['movieId'])
    df_val['userId'] = le_user.transform(df_val['userId'])
    df_val['movieId'] = le_movie.transform(df_val['movieId'])
    
    # Create datasets with negative sampling
    all_movie_ids = df_train['movieId'].unique()
    
    print("Creating datasets...")
    train_dataset = MovieLensDataset(
        users=df_train['userId'].values,
        movies=df_train['movieId'].values,
        ratings=df_train['rating'].values,
        all_movie_ids=all_movie_ids,
        negative_samples=4
    )
    
    val_dataset = MovieLensDataset(
        users=df_val['userId'].values,
        movies=df_val['movieId'].values,
        ratings=df_val['rating'].values,
        all_movie_ids=all_movie_ids,
        negative_samples=4
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    recommendation_model = ImprovedRecommendationModel(
        num_users=len(le_user.classes_),
        num_movies=len(le_movie.classes_),
        embedding_size=256,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2
    ).to(device)
    
    # Training settings
    optimizer = torch.optim.Adam(recommendation_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    loss_func = nn.MSELoss()
    
    # Training loop
    EPOCHS = 10
    best_rmse = float('inf')
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        recommendation_model.train()
        total_loss = 0
        
        for i, batch in enumerate(train_loader):
            users = batch["users"].to(device)
            movies = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)
            
            outputs = recommendation_model(users, movies)
            loss = loss_func(outputs, ratings)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recommendation_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {total_loss/(i+1):.4f}")
        
        # Validation
        print("\nRunning validation...")
        recommendation_model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                users = batch["users"].to(device)
                movies = batch["movies"].to(device)
                ratings = batch["ratings"].to(device)
                
                outputs = recommendation_model(users, movies)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(ratings.cpu().numpy())
        
        # Calculate all metrics
        val_rmse = root_mean_squared_error(val_targets, val_predictions)
        val_mae = mean_absolute_error(val_targets, val_predictions)
        precision, recall = calculate_metrics(
            recommendation_model, 
            val_dataset, 
            all_movie_ids, 
            device
        )
        
        print(f"\nEpoch {epoch} Metrics:")
        print(f"RMSE: {val_rmse:.4f}")
        print(f"MAE: {val_mae:.4f}")
        print(f"Precision@10: {precision:.4f}")
        print(f"Recall@10: {recall:.4f}")
        
        scheduler.step(val_rmse)
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(recommendation_model.state_dict(), 'best_model.pt')
            print(f"Saved new best model with RMSE: {best_rmse:.4f}")
    
    return recommendation_model, le_user, le_movie, movies_df

if __name__ == "__main__":
    model, le_user, le_movie, movies_df = main()