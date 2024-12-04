# Load the datasets
training_data <- read.csv("/Users/ethan/OneDrive/Desktop/KaggleCompetition/train_data.csv")
to_predict_data <- read.csv("/Users/ethan/OneDrive/Desktop/KaggleCompetition/to_predict.csv")

# Extract unique customer IDs
training_customers <- unique(training_data$customer_id)
to_predict_customers <- unique(to_predict_data$customer_id)

# Check if all customers in training.csv are present in to_predict.csv
all_in_to_predict <- all(training_customers %in% to_predict_customers)

# Find customers in training.csv that are NOT in to_predict.csv
missing_from_to_predict <- setdiff(training_customers, to_predict_customers)

# Find customers in to_predict.csv that are NOT in training.csv
extra_in_to_predict <- setdiff(to_predict_customers, training_customers)

# Calculate convergence metrics
total_training_customers <- length(training_customers)
overlap_customers <- length(intersect(training_customers, to_predict_customers))
convergence_rate <- overlap_customers / total_training_customers * 100

# Output the results
if (all_in_to_predict) {
  cat("All customers in 'training.csv' are present in 'to_predict.csv'.\n")
} else {
  cat("Some customers in 'training.csv' are NOT present in 'to_predict.csv'.\n")
  cat("Number of missing customers:", length(missing_from_to_predict), "\n")
}

cat("Number of customers in 'training.csv':", total_training_customers, "\n")
cat("Number of overlapping customers:", overlap_customers, "\n")
cat("Convergence rate:", round(convergence_rate, 2), "%\n")

if (length(extra_in_to_predict) > 0) {
  cat("There are customers in 'to_predict.csv' that are NOT in 'training.csv'.\n")
  cat("Number of extra customers:", length(extra_in_to_predict), "\n")
} else {
  cat("No extra customers in 'to_predict.csv'.\n")
}
