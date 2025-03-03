import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from datetime import timedelta
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.model_selection import KFold


class SequenceProductRecommender:
    def __init__(self, w1=0.4, w2=0.4, w3=0.2, max_sequence_length=3):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.max_sequence_length = max_sequence_length
        self.customer_product_matrix = None
        self.sequence_patterns = defaultdict(Counter)
        self.product_popularity = None
        self.product_id_map = {}
        self.reverse_product_map = {}
        self.customer_id_map = {}
        self.reverse_customer_map = {}

    def _expand_transactions(self, transactions_df):
        if "list_product_id" not in transactions_df.columns:
            return transactions_df

        print("Expanding transactions efficiently...", flush=True)
        transactions_df["product_list"] = (
            transactions_df["list_product_id"].str.strip("[]").str.split()
        )
        expanded_df = transactions_df.explode("product_list").reset_index(drop=True)
        expanded_df["product_id"] = expanded_df["product_list"].astype(int)
        return expanded_df[["time_date", "customer_id", "product_id"]].drop_duplicates()

    def _build_sequence_patterns(self, transactions_df):
        print("Building sequence patterns...", flush=True)
        grouped = transactions_df.sort_values(["customer_id", "time_date"]).groupby(
            "customer_id"
        )

        for customer_id, group in tqdm(grouped, desc="Processing Customer Sequences"):
            product_sequence = group["product_id"].map(self.product_id_map).values
            if len(product_sequence) < 2:
                continue

            for start in range(len(product_sequence) - 1):
                end = min(start + self.max_sequence_length, len(product_sequence) - 1)
                current_seq = tuple(product_sequence[start:end])
                next_product = product_sequence[end]
                self.sequence_patterns[current_seq][next_product] += 1

    def _get_sequence_score(self, customer_sequence, candidate_product):
        total_score, total_weight = 0, 0

        for length in range(
            1, min(self.max_sequence_length + 1, len(customer_sequence) + 1)
        ):
            for i in range(max(0, len(customer_sequence) - length + 1)):
                current_seq = customer_sequence[i : i + length]
                seq_key = tuple(current_seq)

                if seq_key in self.sequence_patterns:
                    total_counts = sum(self.sequence_patterns[seq_key].values())
                    if total_counts > 0:
                        prob = (
                            self.sequence_patterns[seq_key][candidate_product]
                            / total_counts
                        )
                        sequence_weight = length * (i + 1) / len(customer_sequence)
                        total_score += prob * sequence_weight
                        total_weight += sequence_weight

        return total_score / total_weight if total_weight > 0 else 0

    def fit(self, train_df):
        print("Starting model fitting...", flush=True)
        train_df = self._expand_transactions(train_df)

        print("Creating ID mappings...", flush=True)
        self.product_id_map = {
            pid: idx for idx, pid in enumerate(sorted(train_df["product_id"].unique()))
        }
        self.reverse_product_map = {
            idx: pid for pid, idx in self.product_id_map.items()
        }
        self.customer_id_map = {
            cid: idx for idx, cid in enumerate(sorted(train_df["customer_id"].unique()))
        }
        self.reverse_customer_map = {
            idx: cid for cid, idx in self.customer_id_map.items()
        }

        print("Building customer-product matrix...", flush=True)
        customer_idx = train_df["customer_id"].map(self.customer_id_map).values
        product_idx = train_df["product_id"].map(self.product_id_map).values
        data = np.ones(len(customer_idx))
        self.customer_product_matrix = csr_matrix(
            (data, (customer_idx, product_idx)),
            shape=(len(self.customer_id_map), len(self.product_id_map)),
        )

        print("Building sequence patterns...", flush=True)
        self._build_sequence_patterns(train_df)

        print("Calculating product popularity...", flush=True)
        recent_cutoff = train_df["time_date"].max() - timedelta(days=30)
        recent_transactions = train_df[train_df["time_date"] >= recent_cutoff]
        product_counts = recent_transactions["product_id"].value_counts()
        total_transactions = len(recent_transactions)

        self.product_popularity = {
            self.product_id_map[pid]: count / total_transactions
            for pid, count in product_counts.items()
            if pid in self.product_id_map
        }

        print("Model fitting complete.", flush=True)

    def predict(self, customer_id, n_recommendations=12):
        if customer_id not in self.customer_id_map:
            popular_products = sorted(
                self.product_popularity.items(), key=lambda x: x[1], reverse=True
            )
            return [
                self.reverse_product_map[idx]
                for idx, _ in popular_products[:n_recommendations]
            ]

        customer_idx = self.customer_id_map[customer_id]
        similarity_scores = cosine_similarity(
            self.customer_product_matrix[customer_idx], self.customer_product_matrix
        ).flatten()

        customer_transactions = (
            self.customer_product_matrix[customer_idx].toarray().flatten()
        )
        customer_sequence = [
            i for i, val in enumerate(customer_transactions) if val > 0
        ][-self.max_sequence_length :]

        final_scores = np.zeros(self.customer_product_matrix.shape[1])
        for other_customer_idx, similarity in enumerate(similarity_scores):
            if similarity <= 0:
                continue
            final_scores += (
                self.w1
                * similarity
                * self.customer_product_matrix[other_customer_idx].toarray().flatten()
            )

        for product_idx, popularity in self.product_popularity.items():
            final_scores[product_idx] += self.w3 * popularity

        if customer_sequence:
            for product_idx in range(len(final_scores)):
                sequence_score = self._get_sequence_score(
                    customer_sequence, product_idx
                )
                final_scores[product_idx] += self.w2 * sequence_score

        top_product_indices = np.argsort(-final_scores)
        unique_recommendations = []
        seen_products = set()

        for product_idx in top_product_indices:
            if len(unique_recommendations) >= n_recommendations:
                break
            if product_idx not in seen_products:
                unique_recommendations.append(product_idx)
                seen_products.add(product_idx)

        return [self.reverse_product_map[idx] for idx in unique_recommendations]


def k_fold_evaluate(recommender, train_data, val_data, k=5):
    """
    Perform K-Fold cross-validation to evaluate the recommender model, 
    avoiding redundant computations across folds.
    """
    print("\nStarting K-Fold cross-validation...", flush=True)

    # Precompute sequence patterns and other static features from training data
    print("Precomputing static features...", flush=True)
    recommender.fit(train_data)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_data = val_data.reset_index(drop=True)
    map_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(val_data)):
        print(f"\nProcessing Fold {fold + 1}/{k}...", flush=True)

        # Split validation data into training and testing subsets
        train_fold = val_data.iloc[train_idx]
        test_fold = val_data.iloc[test_idx]

        # Combine training data with the current training fold
        combined_train_data = pd.concat([train_data, train_fold], ignore_index=True)

        # Only update the customer-product matrix and popularity
        print("Updating customer-product matrix and popularity...", flush=True)
        customer_idx = combined_train_data["customer_id"].map(recommender.customer_id_map)
        product_idx = combined_train_data["product_id"].map(recommender.product_id_map)

        # Drop rows with invalid mappings
        valid_rows = customer_idx.notna() & product_idx.notna()
        customer_idx = customer_idx[valid_rows].astype(int)
        product_idx = product_idx[valid_rows].astype(int)
        data = np.ones(len(customer_idx))  # Match length of valid rows

        # Update customer-product matrix
        recommender.customer_product_matrix = csr_matrix(
            (data, (customer_idx, product_idx)),
            shape=(len(recommender.customer_id_map), len(recommender.product_id_map)),
        )

        # Update product popularity
        recent_cutoff = combined_train_data["time_date"].max() - timedelta(days=30)
        recent_transactions = combined_train_data[
            combined_train_data["time_date"] >= recent_cutoff
        ]
        product_counts = recent_transactions["product_id"].value_counts()
        total_transactions = len(recent_transactions)

        recommender.product_popularity = {
            recommender.product_id_map[pid]: count / total_transactions
            for pid, count in product_counts.items()
            if pid in recommender.product_id_map
        }

        # Evaluate on the test fold
        fold_map_scores = []
        grouped = test_fold.groupby("customer_id")["product_id"]

        for customer_id in tqdm(grouped.groups.keys(), desc=f"Evaluating Fold {fold + 1}"):
            true_products = set(grouped.get_group(customer_id))
            recommendations = recommender.predict(customer_id, n_recommendations=12)

            # Calculate MAP for this customer
            if true_products:
                precision_at_k = []
                num_relevant = 0
                for k, rec in enumerate(recommendations, 1):
                    if rec in true_products:
                        num_relevant += 1
                        precision_at_k.append(num_relevant / k)

                if precision_at_k:
                    map_score = sum(precision_at_k) / len(true_products)
                    fold_map_scores.append(map_score)

        # Calculate the mean MAP for this fold
        fold_map = np.mean(fold_map_scores) if fold_map_scores else 0
        map_scores.append(fold_map)
        print(f"Fold {fold + 1} MAP: {fold_map:.4f}", flush=True)

    # Return the average MAP across all folds
    mean_map = np.mean(map_scores) if map_scores else 0
    print(f"\nOverall K-Fold MAP: {mean_map:.4f}")
    return mean_map




def optimize_weights(recommender, val_data, num_samples=50, k=5):
    """
    Optimize weights using random sampling and K-Fold evaluation.
    """
    print("\nStarting K-Fold-based weight optimization...", flush=True)
    best_weights = None
    best_score = float("-inf")

    # Generate random weight samples
    for i in range(num_samples):
        weights = np.random.dirichlet([1, 1, 1])  # Ensure weights sum to 1
        recommender.w1, recommender.w2, recommender.w3 = weights

        print(f"\nEvaluating weights: w1={weights[0]:.4f}, w2={weights[1]:.4f}, w3={weights[2]:.4f}")
        map_score = k_fold_evaluate(recommender, train_data, val_data, k=k)

        if map_score > best_score:
            best_score = map_score
            best_weights = weights
            print(f"New best weights found: {weights} -> MAP: {map_score:.4f}", flush=True)

    print("\nBest weights after optimization:")
    print(f"w1 (Similarity): {best_weights[0]:.4f}")
    print(f"w2 (Sequence): {best_weights[1]:.4f}")
    print(f"w3 (Popularity): {best_weights[2]:.4f}")
    print(f"Best K-Fold MAP Score: {best_score:.4f}")

    return best_weights

if __name__ == "__main__":
    # File paths
    train_file = "train_data.csv"
    val_file = "validation_data.csv"
    predict_file = "to_predict.csv"

    # Step 1: Load data
    print("Loading data...", flush=True)
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    to_predict = pd.read_csv(predict_file)

    # Step 2: Filter customers in the prediction set
    predict_customers = set(to_predict["customer_id"])
    train_data = train_data[train_data["customer_id"].isin(predict_customers)]
    val_data = val_data[val_data["customer_id"].isin(predict_customers)]

    # Step 3: Initialize the recommender
    recommender = SequenceProductRecommender(
        w1=0.4, w2=0.4, w3=0.2, max_sequence_length=3
    )

    # Step 4: Expand transactions for `train_data` and `val_data`
    if "list_product_id" in train_data.columns:
        print("Expanding train_data transactions...", flush=True)
        train_data = recommender._expand_transactions(train_data)
    if "list_product_id" in val_data.columns:
        print("Expanding val_data transactions...", flush=True)
        val_data = recommender._expand_transactions(val_data)

    # Step 5: Convert `time_date` to datetime format
    train_data["time_date"] = pd.to_datetime(train_data["time_date"], errors="coerce")
    val_data["time_date"] = pd.to_datetime(val_data["time_date"], errors="coerce")

    # Step 6: Fit the initial model
    print("\nFitting the initial model...", flush=True)
    recommender.fit(train_data)

    # Step 7: Optimize weights using K-Fold validation
    print("\nOptimizing weights using K-Fold validation...", flush=True)
    optimized_weights = optimize_weights(recommender, val_data, num_samples=50, k=5)

    # Step 8: Update recommender weights
    recommender.w1, recommender.w2, recommender.w3 = optimized_weights

    # Step 9: Fit the model with optimized weights on the full dataset
    print("\nFitting the model with optimized weights on the full dataset...", flush=True)
    recommender.fit(train_data)

    # Step 10: Generate predictions for customers in the test set
    print("\nGenerating predictions for test customers...", flush=True)
    predictions = []
    for customer_id in tqdm(predict_customers, desc="Generating Recommendations"):
        recommendations = recommender.predict(customer_id, n_recommendations=12)
        predictions.append(
            {"customer_id": customer_id, "list_product_id": " ".join(map(str, recommendations))}
        )

    # Step 11: Save predictions to the output file
    submission_df = pd.DataFrame(predictions)
    submission_file = "submission.csv"
    submission_df.to_csv(submission_file, index=False)
    print(f"\nSubmission file saved to {submission_file}", flush=True)

    # Step 12: Final evaluation on the full validation dataset
    print("\nFinal evaluation on the full validation dataset...", flush=True)
    k_fold_evaluate(recommender, train_data, val_data, k=1)
