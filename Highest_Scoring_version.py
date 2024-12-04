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
    def __init__(self, w1=0.6, w2=0.4, max_sequence_length=3):
        self.w1 = w1
        self.w2 = w2
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
            final_scores[product_idx] += self.w2 * popularity

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


def kfold_objective(weights, recommender, val_data, train_data, n_splits=5):
    """
    Objective function using K-Fold validation with progress tracking.
    """
    w1, w2 = weights
    recommender.w1, recommender.w2 = w1, w2

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_map_scores = []

    print("Performing K-Fold validation...")
    for fold_idx, (train_idx, val_idx) in tqdm(
        enumerate(kf.split(val_data["customer_id"].unique())),
        desc="K-Fold Iterations",
        total=n_splits,
    ):
        train_customers = val_data["customer_id"].unique()[train_idx]
        val_customers = val_data["customer_id"].unique()[val_idx]

        train_split = train_data[train_data["customer_id"].isin(train_customers)]
        val_split = val_data[val_data["customer_id"].isin(val_customers)]
        val_customers_grouped = val_split.groupby("customer_id")["product_id"]

        recommender.fit(train_split)

        for customer_id in tqdm(
            val_customers_grouped.groups.keys(),
            desc=f"Validating Fold {fold_idx + 1}",
            leave=False,
        ):
            true_products = set(val_customers_grouped.get_group(customer_id))
            if not true_products:
                continue

            recommendations = recommender.predict(customer_id, n_recommendations=12)
            predicted_binary = [1 if rec in true_products else 0 for rec in recommendations]

            num_relevant = 0
            precision_sum = 0
            for k, is_relevant in enumerate(predicted_binary, 1):
                if is_relevant:
                    num_relevant += 1
                    precision_sum += num_relevant / k

            if num_relevant > 0:
                map_score = precision_sum / len(true_products)
                all_map_scores.append(map_score)

    val_map = np.mean(all_map_scores) if all_map_scores else 0
    print(f"Weights {weights} -> MAP: {val_map:.4f} (using K-Fold validation)")
    return -val_map


def optimize_weights_with_kfold(
    recommender, train_data, val_data, n_splits=5, num_samples=100
):
    """
    Optimize weights using K-Fold validation and progress tracking.
    """
    print("\nStarting K-Fold-based weight optimization...")
    best_weights = None
    best_score = float("-inf")

    weight_bounds = [(0.0, 1.0), (0.0, 1.0)]
    constraints = [{"type": "eq", "fun": lambda w: 1 - sum(w)}]

    print(f"Evaluating {num_samples} random weight samples...")
    weight_samples = [
        np.random.dirichlet(np.ones(2)) for _ in range(num_samples)
    ]

    for weights in tqdm(weight_samples, desc="Evaluating Weight Samples"):
        score = -kfold_objective(weights, recommender, val_data, train_data, n_splits)
        if score > best_score:
            best_score = score
            best_weights = weights

    print("\nPerforming final refinement using the best weights...")
    result = minimize(
        fun=kfold_objective,
        x0=best_weights,
        args=(recommender, val_data, train_data, n_splits),
        method="SLSQP",
        bounds=weight_bounds,
        constraints=constraints,
        options={"maxiter": 20, "ftol": 1e-4},
    )

    if result.success and (-result.fun > best_score):
        best_score = -result.fun
        best_weights = result.x

    print("\nBest weights after optimization:")
    print(f"w1 (Similarity): {best_weights[0]:.4f}")
    print(f"w2 (Popularity): {best_weights[1]:.4f}")
    print(f"Best MAP Score: {best_score:.4f}")

    return best_weights

if __name__ == "__main__":
    # File paths
    train_file = "train_data.csv"
    val_file = "validation_data.csv"
    predict_file = "to_predict.csv"

    # Step 1: Load data
    print("Loading data...")
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    to_predict = pd.read_csv(predict_file)

    # Step 2: Filter customers in the prediction set
    predict_customers = set(to_predict["customer_id"])
    train_data = train_data[train_data["customer_id"].isin(predict_customers)]
    val_data = val_data[val_data["customer_id"].isin(predict_customers)]

    # Step 3: Initialize the recommender
    recommender = SequenceProductRecommender(w1=0.6, w2=0.4, max_sequence_length=3)

    # Step 4: Expand transactions for `train_data` and `val_data`
    if "list_product_id" in train_data.columns:
        print("Expanding train_data transactions...")
        train_data = recommender._expand_transactions(train_data)
    if "list_product_id" in val_data.columns:
        print("Expanding val_data transactions...")
        val_data = recommender._expand_transactions(val_data)

    # Step 5: Convert `time_date` to datetime format
    train_data["time_date"] = pd.to_datetime(train_data["time_date"], errors="coerce")
    val_data["time_date"] = pd.to_datetime(val_data["time_date"], errors="coerce")

    # Step 6: Fit the initial model
    print("Fitting the initial model...")
    recommender.fit(train_data)

    # Step 7: Optimize weights
    print("\nOptimizing weights...")
    val_customers = val_data.groupby("customer_id")["product_id"]
    optimized_weights = optimize_weights_with_kfold(
        recommender, train_data, val_data, n_splits=5, num_samples=50
    )

    # Step 8: Update recommender weights
    recommender.w1, recommender.w2 = optimized_weights

    # Step 9: Fit the model with optimized weights
    print("\nFitting the model with optimized weights on the full dataset...")
    full_data = pd.concat([train_data, val_data], ignore_index=True)
    recommender.fit(full_data)

    # Step 10: Generate predictions for customers in the test set
    print("\nGenerating predictions for test customers...")
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
    print(f"\nSubmission file saved to {submission_file}")

    # Step 12: Evaluate on the validation set using the full dataset
    print("\nFinal evaluation on the full validation dataset...")
    true_labels = []
    predicted_labels = []
    all_map_scores = []

    for customer_id in tqdm(val_customers.groups.keys(), desc="Evaluating Validation Data"):
        true_products = set(val_customers.get_group(customer_id))
        recommendations = recommender.predict(customer_id, n_recommendations=12)

        predicted_binary = [1 if rec in true_products else 0 for rec in recommendations]
        true_binary = [1] * len(true_products) + [0] * (12 - len(true_products))

        true_labels.extend(true_binary[:12])
        predicted_labels.extend(predicted_binary[:12])

        if true_products:
            precision_at_k = []
            num_relevant = 0
            for k, is_relevant in enumerate(predicted_binary, 1):
                if is_relevant:
                    num_relevant += 1
                    precision_at_k.append(num_relevant / k)
            if precision_at_k:
                map_score = sum(precision_at_k) / len(true_products)
                all_map_scores.append(map_score)

    # Step 13: Calculate global metrics
    val_accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
    val_precision = np.sum(np.array(true_labels) * np.array(predicted_labels)) / (
        np.sum(predicted_labels) + 1e-9
    )
    val_recall = np.sum(np.array(true_labels) * np.array(predicted_labels)) / (
        np.sum(true_labels) + 1e-9
    )
    val_map = np.mean(all_map_scores) if all_map_scores else 0

    # Step 14: Print the final summary
    print("\nFinal Summary:")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation Mean Average Precision (MAP@12): {val_map:.4f}")