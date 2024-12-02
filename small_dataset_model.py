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
from contextlib import contextmanager

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

class ExplainableRecommenderSystem(DeepRecommenderSystem):
    def __init__(self, num_users, num_movies, embedding_size=128):
        super(ExplainableRecommenderSystem, self).__init__(num_users, num_movies, embedding_size)
        
        # Add attention layer
        self.attention = nn.Sequential(
            nn.Linear(2 * embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # Change to 2 outputs for user and movie attention
            nn.Softmax(dim=1)
        )
    
    def forward(self, users, movies):
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)
        concatenated = torch.cat([user_embedded, movie_embedded], dim=1)
        
        # Calculate attention weights (2 weights per sample)
        attention_weights = self.attention(concatenated)
        
        # Split embeddings and apply attention
        user_weighted = user_embedded * attention_weights[:, 0].unsqueeze(1)
        movie_weighted = movie_embedded * attention_weights[:, 1].unsqueeze(1)
        
        # Concatenate weighted embeddings
        weighted_features = torch.cat([user_weighted, movie_weighted], dim=1)
        
        return self.layers(weighted_features).squeeze(), attention_weights

def explain_recommendation(model, user_id, movie_id, device):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
        movie_tensor = torch.tensor([movie_id], dtype=torch.long).to(device)
        
        prediction, attention = model(user_tensor, movie_tensor)
        
        # Get embedding representations
        user_embedding = model.user_embedding(user_tensor)
        movie_embedding = model.movie_embedding(movie_tensor)
        
        # Generate explanation text
        explanation = (
            f"Recommendation strength: {prediction.item():.2f}\n"
            f"Main factors in this recommendation:\n"
            f"- User preferences: {attention[0, 0].item():.2f}\n"
            f"- Movie characteristics: {attention[0, 1].item():.2f}\n"
        )
        
        return explanation

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            users = batch["users"].to(device)
            movies = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)
            
            optimizer.zero_grad()
            
            # Handle tuple return from ExplainableRecommenderSystem
            output = model(users, movies)
            if isinstance(output, tuple):
                predictions, _ = output  # Unpack predictions and attention weights
            else:
                predictions = output
                
            # Scale predictions to 0-5 range
            scaled_predictions = predictions * 5.0
            loss = criterion(scaled_predictions, ratings)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 100 == 0:
                avg_loss = total_loss / batch_count
                sys.stderr.write(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Avg Loss: {avg_loss:.6f}")
                sys.stderr.flush()
        
        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                users = batch["users"].to(device)
                movies = batch["movies"].to(device)
                ratings = batch["ratings"].to(device)
                
                output = model(users, movies)
                if isinstance(output, tuple):
                    predictions, _ = output
                else:
                    predictions = output
                    
                scaled_predictions = predictions * 5.0
                loss = criterion(scaled_predictions, ratings)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses

def calculate_ndcg(true_ratings, predicted_ratings, k=10):
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

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

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
            # Handle tuple output from ExplainableRecommenderSystem
            if isinstance(output, tuple):
                predictions_batch, _ = output  # Unpack predictions and attention weights
            else:
                predictions_batch = output
                
            scaled_output = predictions_batch * 5.0  # Scale back to 0-5 range
            
            predictions.extend(scaled_output.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())
            
            # Store both predicted and true ratings
            for user, movie, pred, true in zip(users.cpu().numpy(), 
                                             movies.cpu().numpy(),
                                             scaled_output.cpu().numpy(),
                                             ratings.cpu().numpy()):
                user_test_items[user].add(movie)
                user_recommendations[user].append((movie, pred, true))
    
    # Calculate RMSE and MAE
    rmse = root_mean_squared_error(actuals, predictions)
    mae = calculate_mae(actuals, predictions)
    
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
    
    return rmse, mae, avg_precision, avg_recall, f_measure, avg_ndcg

def calculate_precision_recall(user_id, test_items, recommended_items, k=10):
    recommended_k = recommended_items[:k]
    
    # Count how many recommended items are in test set
    hits = len(set(recommended_k) & set(test_items))
    
    # Precision = hits / k
    precision = hits / k if k > 0 else 0
    
    # Recall = hits / total test items
    recall = hits / len(test_items) if test_items else 0
    
    return precision, recall

def recommend_movies(model, user_id, movie_ids, df_movies, device, movie_encoder, top_k=10):
    model.eval()
    
    # Create tensors for prediction
    user_tensor = torch.tensor([user_id] * len(movie_ids), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(movie_ids, dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(user_tensor, movie_tensor)
        if isinstance(output, tuple):
            predictions, _ = output
        else:
            predictions = output
            
        predictions = predictions.cpu().numpy() * 5.0
    
    # Create movie recommendations
    movie_preds = list(zip(movie_ids, predictions))
    movie_preds.sort(key=lambda x: x[1], reverse=True)
    top_movies = movie_preds[:top_k]
    
    # Get movie details
    recommended_movies = []
    for encoded_movie_id, pred_rating in top_movies:
        original_movie_id = movie_encoder.inverse_transform([encoded_movie_id])[0]
        movie_info = df_movies[df_movies['movieId'] == original_movie_id]
        if not movie_info.empty:
            movie_info = movie_info.iloc[0]
            recommended_movies.append({
                'title': movie_info['title'],
                'predicted_rating': pred_rating,
                'movieId': encoded_movie_id  # Add encoded movieId
            })
    
    return recommended_movies

@contextmanager
def stdout_to_file(filename):
    # Context manager to redirect stdout and stderr to both file and console
    class MultiOutputStream:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, text):
            for stream in self.streams:
                stream.write(text)
                stream.flush()

        def flush(self):
            # Fix: Iterate through streams to flush each one
            for stream in self.streams:
                stream.flush()

    with open(filename, 'w') as file:
        stdout_backup = sys.stdout
        stderr_backup = sys.stderr
        multi_stream = MultiOutputStream(file, sys.stdout)
        sys.stdout = multi_stream
        sys.stderr = multi_stream
        try:
            yield
        finally:
            sys.stdout = stdout_backup
            sys.stderr = stderr_backup

def visualize_embeddings(model, users, movies, device):
    # Create t-SNE visualization of user and movie embeddings
    from sklearn.manifold import TSNE
    import seaborn as sns
    
    model.eval()
    with torch.no_grad():
        # Get embeddings
        user_embeddings = model.user_embedding(torch.tensor(users).to(device)).cpu().numpy()
        movie_embeddings = model.movie_embedding(torch.tensor(movies).to(device)).cpu().numpy()
        
        # Combine embeddings
        all_embeddings = np.vstack([user_embeddings, movie_embeddings])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=embeddings_2d[:len(users), 0],
            y=embeddings_2d[:len(users), 1],
            label='Users'
        )
        sns.scatterplot(
            x=embeddings_2d[len(users):, 0],
            y=embeddings_2d[len(users):, 1],
            label='Movies'
        )
        plt.title('User and Movie Embeddings Visualization')
        plt.savefig('embeddings_visualization.png')
        plt.close()

def main():
    with stdout_to_file('small_data_recommendation.txt'):
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load and preprocess data
        print("Loading data...")
        ratings_df = pd.read_csv("data/ratings.csv")
        movies_df = pd.read_csv("data/movies.csv")
        
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
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Initialize explainable model instead
        model = ExplainableRecommenderSystem(
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
        rmse, mae, precision, recall, f_measure, ndcg = calculate_metrics(model, val_loader, device)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
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
        
        # Generate explanations for recommendations
        print("\nGenerating explanations for recommendations...")
        sample_user_id = 1
        recommendations = recommend_movies(model, sample_user_id, movie_ids, movies_df, device, movie_encoder)
        
        for i, movie in enumerate(recommendations[:3], 1):
            explanation = explain_recommendation(
                model,
                sample_user_id,
                movie['movieId'],  # Now this key exists
                device
            )
            print(f"\nExplanation for recommendation {i}:")
            print(explanation)
        
        # Visualize embeddings
        print("\nGenerating embedding visualization...")
        visualize_embeddings(
            model,
            ratings_df['userId'].unique()[:100],  # Sample 100 users
            ratings_df['movieId'].unique()[:100],  # Sample 100 movies
            device
        )

if __name__ == "__main__":
    main()