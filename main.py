import torch
import pandas as pd
df = pd.read_csv("databases/ml-latest-small/ratings.csv")
df.head()
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    """
    The Movie Lens Dataset class. This class prepares the dataset for training and validation.
    """
    def __init__(self, users, movies, ratings):
        """
        Initializes the dataset object with user, movie, and rating data.
        """
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.users)

    def __getitem__(self, item):
        """
        Retrieves a sample from the dataset at the specified index.
        """
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
from sklearn.metrics import mean_squared_error
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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn

class SVDRecommender(nn.Module):
    def __init__(self, num_users, num_items, num_factors=100):
        super(SVDRecommender, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, user_ids, item_ids):
        user_vec = self.user_factors(user_ids)
        item_vec = self.item_factors(item_ids)
        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()
        dot = (user_vec * item_vec).sum(1)
        return dot + user_bias + item_bias + self.global_bias

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device="cuda"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            user_ids = batch["users"].to(device)
            movie_ids = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)
            
            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch["users"].to(device)
                movie_ids = batch["movies"].to(device)
                ratings = batch["ratings"].to(device)
                
                predictions = model(user_ids, movie_ids)
                val_loss += criterion(predictions, ratings).item()
                
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {total_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

def get_top_k_recommendations(model, user_id, train_data, num_items, k=10, device="cuda"):
    model.eval()
    # Get items the user hasn't rated
    user_items = set(train_data[train_data['userId'] == user_id]['movieId'])
    all_items = set(range(num_items))
    items_to_predict = list(all_items - user_items)
    
    # Predict ratings for all unrated items
    with torch.no_grad():
        user_tensor = torch.LongTensor([user_id] * len(items_to_predict)).to(device)
        item_tensor = torch.LongTensor(items_to_predict).to(device)
        predictions = model(user_tensor, item_tensor)
        
    # Get top k items
    _, indices = torch.topk(predictions, k)
    return [items_to_predict[idx] for idx in indices.cpu().numpy()]

# Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SVDRecommender(num_users=len(le_user.classes_), 
                      num_items=len(le_movie.classes_)).to(device)

# Train the model
train_model(model, train_loader, val_loader, epochs=20, device=device)

# Evaluate
model.eval()
predictions = []
true_ratings = []

with torch.no_grad():
    for batch in val_loader:
        user_ids = batch["users"].to(device)
        movie_ids = batch["movies"].to(device)
        ratings = batch["ratings"].to(device)
        
        pred = model(user_ids, movie_ids)
        predictions.extend(pred.cpu().numpy())
        true_ratings.extend(ratings.cpu().numpy())

# Calculate metrics
mae = mean_absolute_error(true_ratings, predictions)
rmse = mean_squared_error(true_ratings, predictions, squared=False)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Generate recommendations for a sample user
sample_user_id = 0
top_10_movies = get_top_k_recommendations(model, sample_user_id, df_train, 
                                        len(le_movie.classes_), k=10, device=device)
print(f"Top 10 recommendations for user {sample_user_id}:")
print(top_10_movies)
