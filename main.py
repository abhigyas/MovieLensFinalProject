import torch
import pandas as pd
import os
df = pd.read_csv("databases/ml-latest-small/ratings.csv")
df.head()
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    
    # The Movie Lens Dataset class. This class prepares the dataset for training and validation.
    
    def __init__(self, users, movies, ratings):
        # Initializes the dataset object with user, movie, and rating data.
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        # Returns the total number of samples in the dataset.
        return len(self.users)

    def __getitem__(self, item):
       
        # Retrieves a sample from the dataset at the specified index.
        
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]

        return {
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float),
        }
import numpy as np
import torch.nn as nn
class RecommendationSystemModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_size=256,
        hidden_dim=256,
        dropout_rate=0.2,
    ):
        super(RecommendationSystemModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_size
        )
        self.movie_embedding = nn.Embedding(
            num_embeddings=self.num_movies, embedding_dim=self.embedding_size
        )

        # Hidden layers
        self.fc1 = nn.Linear(2 * self.embedding_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, users, movies):
        # Embeddings
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)

        # Concatenate user and movie embeddings
        combined = torch.cat([user_embedded, movie_embedded], dim=1)

        # Pass through hidden layers with ReLU activation and dropout
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)

        return output
from sklearn import model_selection

df_train, df_val = model_selection.train_test_split(
    df, test_size=0.1, random_state=3, stratify=df.rating.values
)
from torch.utils.data import DataLoader

BATCH_SIZE = 32

train_loader = DataLoader(df_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = DataLoader(df_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
#

import sys
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming df_train and df_val are already defined
le_user = LabelEncoder()
le_movie = LabelEncoder()

# Combine userId and movieId columns from both training and validation datasets
combined_user_ids = pd.concat([df_train['userId'], df_val['userId']])
combined_movie_ids = pd.concat([df_train['movieId'], df_val['movieId']])

# Fit the LabelEncoder on the combined data
le_user.fit(combined_user_ids)
le_movie.fit(combined_movie_ids)

# Transform the userId and movieId columns in both datasets
df_train['userId'] = le_user.transform(df_train['userId'])
df_train['movieId'] = le_movie.transform(df_train['movieId'])
df_val['userId'] = le_user.transform(df_val['userId'])
df_val['movieId'] = le_movie.transform(df_val['movieId'])

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

recommendation_model = RecommendationSystemModel(
    num_users=len(le_user.classes_), 
    num_movies=len(le_movie.classes_),
    embedding_size=128,
    hidden_dim=256,
    dropout_rate=0.1,
).to(device)

optimizer = torch.optim.Adam(recommendation_model.parameters(), lr=1e-3)
loss_func = nn.MSELoss()

EPOCHS = 10

# Function to log progress
def log_progress(epoch, step, total_loss, log_progress_step, data_size, losses):
    avg_loss = total_loss / log_progress_step
    sys.stderr.write(f"\rEpoch {epoch}, Step {step}/{data_size}, Loss: {avg_loss:.6f}")
    sys.stderr.flush()

total_loss = 0
log_progress_step = 100
losses = []
train_dataset_size = len(train_dataset)
print(f"Training on {train_dataset_size} samples...")

recommendation_model.train()
for e in range(EPOCHS):
    step_count = 0  # Reset step count at the beginning of each epoch
    for i, train_data in enumerate(train_loader):
        output = recommendation_model(
            train_data["users"].to(device), train_data["movies"].to(device)
        )
        # Reshape the model output to match the target's shape
        output = output.squeeze()  # Removes the singleton dimension
        ratings = (
            train_data["ratings"].to(torch.float32).to(device)
        )  # Assuming ratings is already 1D

        loss = loss_func(output, ratings)
        total_loss += loss.sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Increment step count by the actual size of the batch
        step_count += len(train_data["users"])

        # Check if it's time to log progress
        if (
            step_count % log_progress_step == 0 or i == len(train_loader) - 1
        ):  # Log at the end of each epoch
            log_progress(
                e, step_count, total_loss, log_progress_step, train_dataset_size, losses
            )
            total_loss = 0
    print()  # Move to the next line after each epoch

# Evaluation
y_pred = []
y_true = []

recommendation_model.eval()

with torch.no_grad():
    for i, valid_data in enumerate(val_loader):
        output = recommendation_model(
            valid_data["users"].to(device), valid_data["movies"].to(device)
        )
        ratings = valid_data["ratings"].to(device)
        y_pred.extend(output.cpu().numpy())
        y_true.extend(ratings.cpu().numpy())

# Calculate RMSE
rms = root_mean_squared_error(y_true, y_pred)
print(f"RMSE: {rms:.4f}")

def get_unwatched_movies(user_id, df_train):
    user_rated_movies = set(df_train[df_train['userId'] == user_id]['movieId'].values)
    all_movies = set(df_train['movieId'].unique())
    return list(all_movies - user_rated_movies)

def get_top_n_recommendations(model, user_id, n=10, df_train=df_train):
    model.eval()
    with torch.no_grad():
        # Get unwatched movies
        unwatched_movies = get_unwatched_movies(user_id, df_train)
        if len(unwatched_movies) == 0:
            return []
        
        # Convert to tensor format
        user_tensor = torch.tensor([user_id] * len(unwatched_movies)).to(device)
        movie_tensor = torch.tensor(unwatched_movies).to(device)
        
        # Get predictions
        predictions = model(user_tensor, movie_tensor)
        predictions = predictions.squeeze().cpu().numpy()
        
        # Get top N items
        top_n_indices = np.argsort(predictions)[-n:][::-1]
        recommended_movies = [unwatched_movies[i] for i in top_n_indices]
        
        return recommended_movies

def calculate_precision_recall(recommendations, test_items):
    if len(recommendations) == 0 or len(test_items) == 0:
        return 0.0, 0.0
    
    true_positives = len(set(recommendations) & set(test_items))
    precision = true_positives / len(recommendations)
    recall = true_positives / len(test_items)
    
    return precision, recall

def calculate_ndcg(recommendations, test_items, k=10):
    recommendations = np.array(recommendations)
    test_items = np.array(test_items)
    if recommendations.size == 0 or test_items.size == 0:
        return 0.0
    
    relevance = np.zeros(k)
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            relevance[i] = 1
            
    # Calculate DCG
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
    
    # Calculate IDCG
    ideal_relevance = np.zeros(k)
    ideal_relevance[:min(k, len(test_items))] = 1
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    
    return dcg / idcg if idcg > 0 else 0.0

# Example usage
def evaluate_recommendations(model, df_train, df_val, n=10):
    precisions = []
    recalls = []
    ndcgs = []
    
    unique_users = df_val['userId'].unique()
    
    for user_id in unique_users:
        # Get recommendations
        recommendations = get_top_n_recommendations(model, user_id, n, df_train)
        
        # Get ground truth from validation set
        test_items = df_val[df_val['userId'] == user_id]['movieId'].values
        
        # Calculate metrics
        precision, recall = calculate_precision_recall(recommendations, test_items)
        ndcg = calculate_ndcg(recommendations, test_items, k=n)
        
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
    
    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'ndcg': np.mean(ndcgs)
    }

# Run evaluation
metrics = evaluate_recommendations(recommendation_model, df_train, df_val)
print(f"\nEvaluation Metrics:")
print(f"Precision@10: {metrics['precision']:.4f}")
print(f"Recall@10: {metrics['recall']:.4f}")
print(f"NDCG@10: {metrics['ndcg']:.4f}")

# Load movies dataset
movies_df = pd.read_csv("databases/ml-latest-small/movies.csv")

def get_movie_titles(movie_ids, movies_df=movies_df):
    # Convert encoded IDs back to original movie IDs
    original_movie_ids = le_movie.inverse_transform(movie_ids)
    # Get movie titles
    movie_info = movies_df[movies_df['movieId'].isin(original_movie_ids)]
    return movie_info[['movieId', 'title', 'genres']].values.tolist()

# Example: Get recommendations for a sample user
sample_user_id = df_train['userId'].iloc[0]  # Get first user as example
print(f"\nGetting recommendations for user {sample_user_id}:")

# Get top 10 recommendations
recommended_movie_ids = get_top_n_recommendations(recommendation_model, sample_user_id, n=10)
recommended_movies = get_movie_titles(recommended_movie_ids)

print("\nTop 10 Recommended Movies:")
print("-" * 80)
for i, (movie_id, title, genres) in enumerate(recommended_movies, 1):
    print(f"{i}. {title} ({genres})")