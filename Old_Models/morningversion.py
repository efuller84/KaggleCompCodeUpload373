import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from datetime import timedelta
from tqdm import tqdm
from scipy.optimize import minimize


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

        if customer_sequence:
            for product_idx in range(len(final_scores)):
                sequence_score = self._get_sequence_score(
                    customer_sequence, product_idx
                )
                final_scores[product_idx] += self.w2 * sequence_score

        for product_idx, popularity in self.product_popularity.items():
            final_scores[product_idx] += self.w3 * popularity

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

    
def objective(weights, recommender, val_customers, sample_size=500):
    """
    Objective function with optimized sampling and progress tracking for weight optimization.
    """
    w1, w2, w3 = weights
    recommender.w1, recommender.w2, recommender.w3 = w1, w2, w3

    # Stratified sampling based on customer activity levels
    customer_frequencies = val_customers.size().sort_values(ascending=False)

    # Split customers into three activity levels
    high_activity = customer_frequencies.head(len(customer_frequencies) // 3).index
    med_activity = customer_frequencies.iloc[len(customer_frequencies) // 3: 2 * len(customer_frequencies) // 3].index
    low_activity = customer_frequencies.tail(len(customer_frequencies) // 3).index

    # Sample proportionally from each group with error handling
    sampled_high = np.random.choice(high_activity, size=min(sample_size // 3, len(high_activity)), replace=False)
    sampled_med = np.random.choice(med_activity, size=min(sample_size // 3, len(med_activity)), replace=False)
    sampled_low = np.random.choice(low_activity, size=min(sample_size // 3, len(low_activity)), replace=False)

    sampled_customers = np.concatenate([sampled_high, sampled_med, sampled_low])

    all_map_scores = []
    batch_size = 50  # Process in smaller batches

    print(f"Evaluating {len(sampled_customers)} sampled customers...")

    for batch_start in tqdm(range(0, len(sampled_customers), batch_size), desc="Processing batches"):
        batch_customers = sampled_customers[batch_start: batch_start + batch_size]

        for customer_id in batch_customers:
            try:
                true_products = set(val_customers.get_group(customer_id))
                if not true_products:
                    continue

                recommendations = recommender.predict(customer_id, n_recommendations=12)
                predicted_binary = [1 if rec in true_products else 0 for rec in recommendations]

                # Fast MAP calculation
                num_relevant = 0
                precision_sum = 0
                for k, is_relevant in enumerate(predicted_binary, 1):
                    if is_relevant:
                        num_relevant += 1
                        precision_sum += num_relevant / k

                if num_relevant > 0:
                    map_score = precision_sum / len(true_products)
                    all_map_scores.append(map_score)

            except Exception as e:
                print(f"Error processing customer {customer_id}: {str(e)}")
                continue

    val_map = np.mean(all_map_scores) if all_map_scores else 0
    print(f"Weights {weights} -> MAP: {val_map:.4f} (on {len(all_map_scores)} valid samples)", flush=True)
    return -val_map  # Negative because we're minimizing


def optimize_weights(recommender, train_data, val_customers):
    """
    Optimized weight search with efficient sampling and logging.
    """
    starting_points = [
        [0.4, 0.4, 0.2],  # Default weights
        [0.33, 0.33, 0.34]  # Balanced weights
    ]

    constraints = [{'type': 'eq', 'fun': lambda w: 1 - sum(w)}]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    best_result = None
    best_score = float('-inf')

    print("Starting weight optimization...")

    for start in starting_points:
        print(f"\nTesting starting point: {start}")
        result = minimize(
            fun=objective,
            x0=start,
            args=(recommender, val_customers, 500),  # Sample size for objective
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 20,  # Balanced tradeoff between speed and accuracy
                'ftol': 1e-4,
                'disp': True
            }
        )

        if result.success and (-result.fun > best_score):
            best_score = -result.fun
            best_result = result

    final_weights = best_result.x if best_result else np.array([0.4, 0.4, 0.2])
    final_weights = final_weights / np.sum(final_weights)

    # Validate final weights with a larger sample
    print("\nValidating final weights...")
    final_score = -objective(final_weights, recommender, val_customers, sample_size=1000)

    print("\nFinal Optimization Results:")
    print(f"w1 (Similarity): {final_weights[0]:.4f}")
    print(f"w2 (Sequence): {final_weights[1]:.4f}")
    print(f"w3 (Popularity): {final_weights[2]:.4f}")
    print(f"Validated MAP score: {final_score:.4f}", flush=True)

    return final_weights



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
    recommender = SequenceProductRecommender(
        w1=0.4, w2=0.4, w3=0.2, max_sequence_length=3
    )

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
    optimized_weights = optimize_weights(recommender, train_data, val_customers)

    # Step 8: Update recommender weights
    recommender.w1, recommender.w2, recommender.w3 = optimized_weights

    # Step 9: Fit the model with optimized weights
    print("\nFitting the model with optimized weights...")
    recommender.fit(train_data)

    # Step 10: Evaluate on the validation set
    print("\nEvaluating on the validation set...")
    true_labels = []
    predicted_labels = []
    all_map_scores = []

    for customer_id in tqdm(predict_customers, desc="Evaluating Validation Data"):
        if customer_id not in val_customers.groups:
            continue
        true_products = set(val_customers.get_group(customer_id))
        recommendations = recommender.predict(customer_id, n_recommendations=12)

        # Binary classification for metrics
        predicted_binary = [1 if rec in true_products else 0 for rec in recommendations]
        true_binary = [1] * len(true_products) + [0] * (12 - len(true_products))

        # Append metrics
        true_labels.extend(true_binary[:12])
        predicted_labels.extend(predicted_binary[:12])

        # Calculate MAP score for this customer
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

    # Step 11: Calculate global metrics
    val_accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
    val_precision = np.sum(np.array(true_labels) * np.array(predicted_labels)) / (
        np.sum(predicted_labels) + 1e-9
    )
    val_recall = np.sum(np.array(true_labels) * np.array(predicted_labels)) / (
        np.sum(true_labels) + 1e-9
    )
    val_map = np.mean(all_map_scores) if all_map_scores else 0

    # Step 12: Print the final summary
    print("\nFinal Summary:")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation Mean Average Precision: {val_map:.4f}")
