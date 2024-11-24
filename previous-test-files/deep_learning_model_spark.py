import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer

def calculate_rmse(pred, true):
    # Squeeze pred to match true's dimensions
    pred = pred.squeeze()
    return torch.sqrt(nn.MSELoss()(pred, true)).item()

def calculate_mae(pred, true):
    # Squeeze pred to match true's dimensions
    pred = pred.squeeze()
    return nn.L1Loss()(pred, true).item()

def calculate_precision_recall(interaction_pred, label_true, threshold=0.5):
    pred_labels = (interaction_pred.squeeze() >= threshold).cpu().numpy()
    true_labels = label_true.cpu().numpy()
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    return precision, recall

def calculate_f_measure(precision, recall):
    """Calculate F1 score from precision and recall"""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def calculate_ndcg(predictions, true_items, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain
    Args:
        predictions: List of predicted item IDs
        true_items: List of true relevant item IDs
        k: Number of items to consider
    """
    dcg = 0
    idcg = 0
    
    # Calculate DCG
    for i, item_id in enumerate(predictions[:k]):
        if item_id in true_items:
            rel = 1
            dcg += rel / np.log2(i + 2)
    
    # Calculate IDCG
    for i in range(min(len(true_items), k)):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0

class BalancedMovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings, transform=None):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        self.transform = transform
        self.normalized_ratings = (ratings - 1) / 4.0
        self.labels = (ratings >= 4.0).astype(np.int32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        if self.transform:
            rating = self.transform(self.normalized_ratings[idx])
        else:
            rating = self.normalized_ratings[idx]
        return {
            "users": torch.tensor(self.users[idx], dtype=torch.long),
            "movies": torch.tensor(self.movies[idx], dtype=torch.long),
            "ratings": torch.tensor(rating, dtype=torch.float),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

class ResidualBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return x + self.layer(x)

class DeepRecommenderModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=128, layers=[256, 128, 64], dropout_rate=0.2):
        super(DeepRecommenderModel, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        
        self.mlp_layers = []
        input_dim = embedding_dim * 3  # Changed for element-wise multiplication

        self.residual_layers = nn.ModuleList()
        for layer_dim in layers:
            layer = nn.Sequential(
                nn.Linear(input_dim, layer_dim),
                nn.LayerNorm(layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.residual_layers.append(layer)
            if input_dim == layer_dim:
                self.mlp_layers.append(ResidualBlock(layer))
            else:
                self.mlp_layers.append(layer)
            input_dim = layer_dim

        self.mlp = nn.Sequential(*self.mlp_layers)

        self.rating_predictor = nn.Sequential(
            nn.Linear(layers[-1], layers[-1]//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(layers[-1]//2, 1),
            nn.Sigmoid()
        )

        self.interaction_predictor = nn.Sequential(
            nn.Linear(layers[-1], layers[-1]//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(layers[-1]//2, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, users, movies):
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        
        user_bias = self.user_bias(users)
        movie_bias = self.movie_bias(movies)
        
        element_wise = user_emb * movie_emb
        concat = torch.cat([user_emb, movie_emb, element_wise], dim=1)
        
        features = self.mlp(concat)
        
        rating_pred = self.rating_predictor(features) * 4.0 + 1.0 + user_bias + movie_bias
        interaction_pred = self.interaction_predictor(features)
        
        return rating_pred, interaction_pred

def create_balanced_sampler(dataset):
    labels = dataset.labels
    class_counts = Counter(labels)
    total_samples = len(labels)

    class_weights = {class_id: total_samples / count for class_id, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def custom_loss(rating_pred, interaction_pred, rating_true, label_true, alpha=0.7):
    mse_loss = nn.MSELoss()(rating_pred.squeeze(), rating_true)
    bce_loss = nn.BCELoss()(interaction_pred.squeeze(), label_true)
    mae_loss = nn.L1Loss()(rating_pred.squeeze(), rating_true)
    return alpha * (0.8 * mse_loss + 0.2 * mae_loss) + (1 - alpha) * bce_loss

def evaluate_model(model, dataloader, device):
    model.eval()
    metrics = {'val_loss': 0, 'rmse': 0, 'mae': 0, 
               'precision': 0, 'recall': 0, 'f1': 0, 'ndcg': 0}
    
    with torch.no_grad():
        for batch in dataloader:
            users = batch["users"].to(device)
            movies = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)
            labels = batch["labels"].to(device)

            rating_pred, interaction_pred = model(users, movies)
            
            # Calculate existing metrics
            metrics['val_loss'] += custom_loss(rating_pred, interaction_pred, ratings, labels).item()
            metrics['rmse'] += calculate_rmse(rating_pred, ratings)
            metrics['mae'] += calculate_mae(rating_pred, ratings)
            
            # Calculate new metrics
            prec, rec = calculate_precision_recall(interaction_pred, labels)
            metrics['precision'] += prec
            metrics['recall'] += rec
            metrics['f1'] += calculate_f_measure(prec, rec)

    # Average metrics
    n = len(dataloader)
    return {k: v/n for k, v in metrics.items()}

def train_model(model, train_loader, val_loader, epochs=15, device="cuda"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            users = batch["users"].to(device)
            movies = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            rating_pred, interaction_pred = model(users, movies)
            loss = custom_loss(rating_pred, interaction_pred, ratings, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Get evaluation metrics
        metrics = evaluate_model(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Validation RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        print(f"Validation Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"Validation F1: {metrics['f1']:.4f}, NDCG: {metrics['ndcg']:.4f}")

        if metrics['val_loss'] < best_val_loss:
            best_val_loss = metrics['val_loss']
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def prepare_data(df):
    # Filter users with at least 5 ratings
    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    df = df[df['userId'].isin(valid_users)]

    # Filter movies with at least 3 ratings
    movie_counts = df['movieId'].value_counts()
    valid_movies = movie_counts[movie_counts >= 3].index
    df = df[df['movieId'].isin(valid_movies)]

    # Add user and movie average ratings
    user_avg = df.groupby('userId')['rating'].mean()
    movie_avg = df.groupby('movieId')['rating'].mean()
    df['user_avg_rating'] = df['userId'].map(user_avg)
    df['movie_avg_rating'] = df['movieId'].map(movie_avg)

    # Balance the dataset
    positive_samples = df[df['rating'] >= 4.0]
    negative_samples = df[df['rating'] < 4.0]

    if len(negative_samples) > 1.5 * len(positive_samples):
        negative_samples = negative_samples.sample(
            n=int(1.5 * len(positive_samples)),
            random_state=42
        )

    balanced_df = pd.concat([positive_samples, negative_samples])
    return balanced_df

def get_predictions_for_user(user_id, model_path, df, le_user, le_movie):
    """
    Get predictions for a specific user using a trained model
    Args:
        user_id: User ID to get predictions for
        model_path: Path to saved model weights
        df: DataFrame containing movie data
        le_user: Fitted LabelEncoder for users
        le_movie: Fitted LabelEncoder for movies
    Returns:
        DataFrame with movie predictions
    """
    # Initialize model with same architecture as training
    model = DeepRecommenderModel(
        num_users=len(le_user.classes_),
        num_movies=len(le_movie.classes_),
        embedding_dim=128
    )
    
    # Load saved model weights
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    # Move model to same device as training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Get unique movies
    movies = df['movieId'].unique()
    
    # Transform IDs using label encoders
    encoded_user_id = le_user.transform([user_id])[0]
    encoded_movies = le_movie.transform(movies)
    
    # Create tensors for prediction
    user_ids = torch.full((len(movies),), encoded_user_id)
    movie_ids = torch.tensor(encoded_movies)
    
    # Get predictions
    with torch.no_grad():
        rating_pred, _ = model(user_ids.to(device), movie_ids.to(device))
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'movieId': movies,
        'predicted_rating': rating_pred.cpu().numpy().squeeze()
    })
    
    return results_df.sort_values('predicted_rating', ascending=False)

def get_top_n_recommendations(model, user_id, movie_ids, n=10, device='cuda'):
    
    # Get top N movie recommendations for a user
    # Args:
    #     model: Trained model
    #     user_id: Encoded user ID
    #     movie_ids: List of encoded movie IDs
    #     n: Number of recommendations
    #     device: Computing device
    
    model.eval()
    with torch.no_grad():
        # Create tensors for all movies for this user
        users = torch.full((len(movie_ids),), user_id, device=device)
        movies = torch.tensor(movie_ids, device=device)
        
        # Get predictions
        rating_preds, _ = model(users, movies)
        
        # Get top N movies
        top_n_indices = rating_preds.squeeze().argsort(descending=True)[:n]
        top_n_movies = movie_ids[top_n_indices.cpu()]
        top_n_scores = rating_preds.squeeze()[top_n_indices].cpu()
        
        return top_n_movies, top_n_scores

def evaluate_recommendations(model, test_data, movie_ids, k=10, device='cuda'):
    # Evaluate recommendations using multiple metrics
    # Args:
    #     model: Trained model
    #     test_data: Test dataset
    #     movie_ids: List of all movie IDs
    #     k: Number of recommendations to evaluate
    #     device: Computing device
    precision_list = []
    recall_list = []
    f1_list = []
    ndcg_list = []
    
    # Group test data by user
    user_items = {}
    for user, items in test_data.groupby('userId'):
        user_items[user] = set(items[items['rating'] >= 4]['movieId'])
    
    for user_id, true_items in user_items.items():
        # Get recommendations
        rec_items, _ = get_top_n_recommendations(model, user_id, movie_ids, n=k, device=device)
        rec_items = set(rec_items.tolist())
        
        # Calculate metrics
        hits = len(rec_items & true_items)
        precision = hits / k if k > 0 else 0
        recall = hits / len(true_items) if true_items else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(calculate_f_measure(precision, recall))
        ndcg_list.append(calculate_ndcg(list(rec_items), list(true_items), k))
    
    return {
        'precision': np.mean(precision_list),
        'recall': np.mean(recall_list),
        'f1': np.mean(f1_list),
        'ndcg': np.mean(ndcg_list)
    }

def init_spark():
    import os
    import sys
    
    # Ensure SPARK_HOME is set correctly
    spark_home = os.path.expanduser('~/hadoop/lib/spark')
    os.environ['SPARK_HOME'] = spark_home
    os.environ['HADOOP_HOME'] = spark_home
    
    # Add Spark python modules to path
    sys.path.append(f"{spark_home}/python")
    sys.path.append(f"{spark_home}/python/lib/py4j-0.10.9.7-src.zip")
    
    spark = SparkSession.builder \
        .appName("MovieLensRecommender") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.extraLibraryPath", f"{spark_home}/lib/native") \
        .config("spark.executor.extraLibraryPath", f"{spark_home}/lib/native") \
        .config("spark.driver.extraClassPath", f"{spark_home}/jars/*") \
        .config("spark.executor.extraClassPath", f"{spark_home}/jars/*") \
        .getOrCreate()

    # Set log level to ERROR to suppress warnings if needed
    # spark.sparkContext.setLogLevel("ERROR")
    
    return spark

def prepare_data_spark(spark, input_path):

    # Read CSV with Spark
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Calculate user and movie counts using window functions
    user_counts = Window.partitionBy("userId")
    movie_counts = Window.partitionBy("movieId")
    
    df = df.withColumn("user_rating_count", F.count("rating").over(user_counts))
    df = df.withColumn("movie_rating_count", F.count("rating").over(movie_counts))
    
    # Filter based on minimum ratings
    df = df.filter((F.col("user_rating_count") >= 5) & 
                   (F.col("movie_rating_count") >= 3))
    
    # Calculate average ratings
    df = df.withColumn("user_avg_rating", F.avg("rating").over(user_counts))
    df = df.withColumn("movie_avg_rating", F.avg("rating").over(movie_counts))
    
    # Create balanced dataset
    pos_samples = df.filter(F.col("rating") >= 4.0)
    neg_samples = df.filter(F.col("rating") < 4.0)
    
    pos_count = pos_samples.count()
    neg_count = neg_samples.count()
    
    if neg_count > 1.5 * pos_count:
        neg_samples = neg_samples.sample(False, (1.5 * pos_count) / neg_count)
    
    balanced_df = pos_samples.unionAll(neg_samples)
    
    # Index users and movies
    user_indexer = StringIndexer(inputCol="userId", outputCol="userIdEncoded")
    movie_indexer = StringIndexer(inputCol="movieId", outputCol="movieIdEncoded")
    
    balanced_df = user_indexer.fit(balanced_df).transform(balanced_df)
    balanced_df = movie_indexer.fit(balanced_df).transform(balanced_df)
    
    return balanced_df

def convert_spark_to_pytorch(spark_df):
    # Convert Spark DataFrame to PyTorch Dataset
    # Convert to pandas for final PyTorch processing
    pdf = spark_df.toPandas()
    
    dataset = BalancedMovieLensDataset(
        users=pdf['userIdEncoded'].values,
        movies=pdf['movieIdEncoded'].values,
        ratings=pdf['rating'].values
    )
    
    return dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Spark
    spark = init_spark()
    
    # Load and prepare data using Spark
    input_path = "databases/ml-latest-small/ratings.csv"
    spark_df = prepare_data_spark(spark, input_path)
    
    # Split data
    train_df, val_df = spark_df.randomSplit([0.9, 0.1], seed=42)
    
    # Convert to PyTorch datasets
    train_dataset = convert_spark_to_pytorch(train_df)
    val_dataset = convert_spark_to_pytorch(val_df)
    
    # Get number of unique users and movies
    n_users = spark_df.select("userIdEncoded").distinct().count()
    n_movies = spark_df.select("movieIdEncoded").distinct().count()
    
    # Create data loaders
    train_sampler = create_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        sampler=train_sampler,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False,
        num_workers=4
    )
    
    # Initialize and train model
    model = DeepRecommenderModel(
        num_users=n_users,
        num_movies=n_movies,
        embedding_dim=128,
        layers=[256, 128, 64]
    ).to(device)
    
    train_model(model, train_loader, val_loader, epochs=15, device=device)
    
    # Clean up Spark
    spark.stop()

if __name__ == "__main__":
    main()