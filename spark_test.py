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
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, row_number
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F

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

class MovieLensDataset(Dataset):
    def __init__(self, spark_df):
        # Convert Spark DataFrame to Pandas and then to numpy arrays
        pdf = spark_df.toPandas()
        self.users = pdf['userIdEncoded'].values    # Changed from userId to userIdEncoded
        self.movies = pdf['movieIdEncoded'].values  # Changed from movieId to movieIdEncoded
        self.ratings = pdf['rating'].values

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
            predictions = model(users, movies)
            loss = criterion(predictions * 5.0, ratings)  # Scale predictions to 0-5 range
            
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
                
                predictions = model(users, movies)
                loss = criterion(predictions * 5.0, ratings)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses

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
            
            # Store test items and predictions per user
            for user, movie, pred in zip(users.cpu().numpy(), 
                                       movies.cpu().numpy(), 
                                       scaled_output.cpu().numpy()):
                user_test_items[user].add(movie)
                user_recommendations[user].append((movie, pred))
    
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
    
    return rmse, avg_precision, avg_recall, f_measure

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
                'genres': movie_info['genres'],
                'predicted_rating': pred_rating
            })
    
    return recommended_movies

def init_spark():
    spark = SparkSession.builder \
        .appName("MovieRecommender") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    return spark

def prepare_data(spark):
    # Load data
    ratings_df = spark.read.csv("databases/ml-latest-small/ratings.csv", header=True, inferSchema=True)
    movies_df = spark.read.csv("databases/ml-latest-small/movies.csv", header=True, inferSchema=True)

    # Create indexers
    user_indexer = StringIndexer(inputCol="userId", outputCol="userIdEncoded", handleInvalid='error')
    movie_indexer = StringIndexer(inputCol="movieId", outputCol="movieIdEncoded", handleInvalid='error')

    # Fit and transform
    ratings_df = user_indexer.fit(ratings_df).transform(ratings_df)
    ratings_df = movie_indexer.fit(ratings_df).transform(ratings_df)

    # Convert to integers
    ratings_df = ratings_df.withColumn(
        "userIdEncoded", 
        F.col("userIdEncoded").cast("integer")
    )
    ratings_df = ratings_df.withColumn(
        "movieIdEncoded", 
        F.col("movieIdEncoded").cast("integer")
    )

    # Create dense 0-based indices
    user_window = Window.orderBy("userIdEncoded")
    movie_window = Window.orderBy("movieIdEncoded")
    
    ratings_df = ratings_df.withColumn(
        "userIdEncoded",
        F.row_number().over(user_window) - 1
    )
    ratings_df = ratings_df.withColumn(
        "movieIdEncoded",
        F.row_number().over(movie_window) - 1
    )

    # Cache for performance
    ratings_df.cache()

    # Get number of unique users and movies
    user_count = ratings_df.select("userIdEncoded").distinct().count()
    movie_count = ratings_df.select("movieIdEncoded").distinct().count()

    # Get min and max values
    user_range = ratings_df.agg(
        F.min("userIdEncoded").alias("min_user"),
        F.max("userIdEncoded").alias("max_user")
    ).collect()[0]
    
    movie_range = ratings_df.agg(
        F.min("movieIdEncoded").alias("min_movie"),
        F.max("movieIdEncoded").alias("max_movie")
    ).collect()[0]

    print(f"User ID range: {user_range['min_user']} to {user_range['max_user']}")
    print(f"Movie ID range: {movie_range['min_movie']} to {movie_range['max_movie']}")
    print(f"Number of unique users: {user_count}")
    print(f"Number of unique movies: {movie_count}")

    # Split data
    train_df, val_df = ratings_df.randomSplit([0.8, 0.2], seed=42)

    return train_df, val_df, movies_df, user_count, movie_count

def main():
    spark = init_spark()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and preparing data...")
    train_df, val_df, movies_df, num_users, num_movies = prepare_data(spark)
    
    print(f"Creating model with {num_users} users and {num_movies} movies")
    
    # Create datasets with validation
    train_dataset = MovieLensDataset(
        spark_df=train_df.select(
            "userIdEncoded",
            "movieIdEncoded",
            "rating"
        ).where(
            (F.col("userIdEncoded") < num_users) & 
            (F.col("movieIdEncoded") < num_movies)
        )
    )
    
    val_dataset = MovieLensDataset(
        spark_df=val_df.select(
            "userIdEncoded",
            "movieIdEncoded",
            "rating"
        ).where(
            (F.col("userIdEncoded") < num_users) & 
            (F.col("movieIdEncoded") < num_movies)
        )
    )

    # Create model with exact dimensions
    model = DeepRecommenderSystem(
        num_users=num_users,
        num_movies=num_movies
    ).to(device)

    # Use smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
    rmse, precision, recall, f_measure = calculate_metrics(model, val_loader, device)
    print(f"RMSE: {rmse:.4f}")
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")
    print(f"F-measure: {f_measure:.4f}")
    
    # Generate sample recommendations
    print("\nGenerating sample recommendations...")
    sample_user_id = 1
    movie_ids = val_df.select("movieIdEncoded").distinct().rdd.map(lambda row: row[0]).collect()
    recommendations = recommend_movies(model, sample_user_id, movie_ids, movies_df, device, movie_encoder)
    
    print(f"\nTop 10 recommended movies for user {sample_user_id}:")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie['title']} - Predicted rating: {movie['predicted_rating']:.2f}")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()