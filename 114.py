import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from datetime import timedelta
from tqdm import tqdm


class SequenceProductRecommender:
    def __init__(self, w1=0.3, w2=0.4, w3=0.3, max_sequence_length=3):
        self.w1 = w1  # Similarity weight
        self.w2 = w2  # Sequence weight
        self.w3 = w3  # Popularity weight
        self.max_sequence_length = max_sequence_length
        self.customer_product_matrix = None
        self.sequence_patterns = defaultdict(Counter)
        self.product_popularity = None
        self.product_id_map = {}
        self.reverse_product_map = {}
        self.customer_id_map = {}
        self.reverse_customer_map = {}

    def _expand_transactions(self, transactions_df):
        """Expand transactions using vectorized operations."""
        if 'list_product_id' not in transactions_df.columns:
            return transactions_df

        print("Expanding transactions efficiently...", flush=True)
        # Split strings into lists
        transactions_df['product_list'] = transactions_df['list_product_id'].str.strip('[]').str.split()
        
        # Explode the DataFrame so that each product ID gets its own row
        expanded_df = transactions_df.explode('product_list').reset_index(drop=True)
        
        # Convert product IDs to integers
        expanded_df['product_id'] = expanded_df['product_list'].astype(int)
        
        # Return the expanded DataFrame with only relevant columns
        return expanded_df[['time_date', 'customer_id', 'product_id']].drop_duplicates()

    def _build_sequence_patterns(self, transactions_df):
        """Build sequence patterns efficiently."""
        print("Building sequence patterns...", flush=True)
        grouped = transactions_df.sort_values(['customer_id', 'time_date']).groupby('customer_id')

        for customer_id, group in tqdm(grouped, desc="Processing Customer Sequences"):
            product_sequence = group['product_id'].map(self.product_id_map).values
            if len(product_sequence) < 2:
                continue

            for start in range(len(product_sequence) - 1):
                end = min(start + self.max_sequence_length, len(product_sequence) - 1)
                current_seq = tuple(product_sequence[start:end])
                next_product = product_sequence[end]
                self.sequence_patterns[current_seq][next_product] += 1
                
    def _get_sequence_score(self, customer_sequence, candidate_product):
        """Calculate sequence-based score for a candidate product."""
        total_score = 0
        total_weight = 0

        for length in range(1, min(self.max_sequence_length + 1, len(customer_sequence) + 1)):
            for i in range(max(0, len(customer_sequence) - length + 1)):
                current_seq = customer_sequence[i:i + length]
                seq_key = tuple(current_seq)

                if seq_key in self.sequence_patterns:
                    total_counts = sum(self.sequence_patterns[seq_key].values())
                    if total_counts > 0:
                        prob = self.sequence_patterns[seq_key][candidate_product] / total_counts
                        sequence_weight = length * (i + 1) / len(customer_sequence)
                        total_score += prob * sequence_weight
                        total_weight += sequence_weight

        return total_score / total_weight if total_weight > 0 else 0

    def fit(self, train_df):
        """Fit the model with transaction data."""
        print("Starting model fitting...", flush=True)
        train_df = self._expand_transactions(train_df)

        print("Creating ID mappings...", flush=True)
        self.product_id_map = {pid: idx for idx, pid in enumerate(tqdm(sorted(train_df['product_id'].unique()), desc="Mapping Products"))}
        self.reverse_product_map = {idx: pid for pid, idx in self.product_id_map.items()}
        self.customer_id_map = {cid: idx for idx, cid in enumerate(tqdm(sorted(train_df['customer_id'].unique()), desc="Mapping Customers"))}
        self.reverse_customer_map = {idx: cid for cid, idx in self.customer_id_map.items()}

        print("Building customer-product matrix...", flush=True)
        customer_idx = train_df['customer_id'].map(self.customer_id_map).values
        product_idx = train_df['product_id'].map(self.product_id_map).values
        data = np.ones(len(customer_idx))
        self.customer_product_matrix = csr_matrix(
            (data, (customer_idx, product_idx)),
            shape=(len(self.customer_id_map), len(self.product_id_map))
        )

        print("Building sequence patterns...", flush=True)
        self._build_sequence_patterns(train_df)

        print("Calculating product popularity...", flush=True)
        recent_cutoff = train_df['time_date'].max() - timedelta(days=30)
        recent_transactions = train_df[train_df['time_date'] >= recent_cutoff]
        product_counts = recent_transactions['product_id'].value_counts()
        total_transactions = len(recent_transactions)

        self.product_popularity = {
            self.product_id_map[pid]: count / total_transactions
            for pid, count in tqdm(product_counts.items(), desc="Calculating Popularity")
            if pid in self.product_id_map
        }

        print("Model fitting complete.", flush=True)

    def predict(self, customer_id, n_recommendations=12):
        """Generate recommendations using the trained model."""
        if customer_id not in self.customer_id_map:
            # Handle new customers by returning the most popular products
            popular_products = sorted(self.product_popularity.items(), key=lambda x: x[1], reverse=True)
            return [self.reverse_product_map[idx] for idx, _ in popular_products[:n_recommendations]]

        # Get the index of the target customer
        customer_idx = self.customer_id_map[customer_id]
        # Compute cosine similarity between the target customer and all customers
        similarity_scores = cosine_similarity(
            self.customer_product_matrix[customer_idx],
            self.customer_product_matrix
        ).flatten()

        # Accumulate scores for all products based on similar customers
        scores = np.zeros(self.customer_product_matrix.shape[1])
        for other_customer_idx, similarity in enumerate(similarity_scores):
            if similarity <= 0:  # Skip low or negative similarity
                continue
            scores += similarity * self.customer_product_matrix[other_customer_idx].toarray().flatten()

        # Get top product indices by score, ensuring no duplicate recommendations
        top_product_indices = np.argsort(-scores)
        unique_recommendations = []
        seen_products = set()

        for product_idx in top_product_indices:
            if len(unique_recommendations) >= n_recommendations:
                break
            if product_idx not in seen_products:
                unique_recommendations.append(product_idx)
                seen_products.add(product_idx)

        # Map product indices back to product IDs
        return [self.reverse_product_map[idx] for idx in unique_recommendations]


if __name__ == "__main__":
    train_file = "train_data.csv"
    val_file = "validation_data.csv"
    predict_file = "to_predict.csv"

    print("Loading data...")
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    to_predict = pd.read_csv(predict_file)

    # Filter train_data and val_data for customers in to_predict
    predict_customers = set(to_predict['customer_id'])
    train_data = train_data[train_data['customer_id'].isin(predict_customers)]
    val_data = val_data[val_data['customer_id'].isin(predict_customers)]

    recommender = SequenceProductRecommender(w1=0.3, w2=0.4, w3=0.3, max_sequence_length=3)

    if 'list_product_id' in train_data.columns:
        print("Expanding train_data transactions...")
        train_data = recommender._expand_transactions(train_data)
    if 'list_product_id' in val_data.columns:
        print("Expanding val_data transactions...")
        val_data = recommender._expand_transactions(val_data)

    train_data['time_date'] = pd.to_datetime(train_data['time_date'])
    val_data['time_date'] = pd.to_datetime(val_data['time_date'])

    print("Fitting the model...")
    recommender.fit(train_data)

    print("Generating predictions for customers in to_predict.csv...")
    predictions = {}
    for customer_id in tqdm(predict_customers, desc="Generating Predictions"):
        predictions[customer_id] = recommender.predict(customer_id, n_recommendations=12)

    print("Predictions complete. Saving to file...")
    output_df = pd.DataFrame({
        "customer_id": list(predictions.keys()),
        "predicted_products": [",".join(map(str, preds)) for preds in predictions.values()]
    })
    output_df.to_csv("submission_file.csv", index=False)
    print("Predictions saved to submission_file.csv")
