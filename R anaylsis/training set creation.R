# Load necessary libraries
library(dplyr)
library(tidyr)
library(stringr)

# Paths to input datasets
training_file <- "/Users/ethan/OneDrive/Desktop/KaggleCompetition/train_data.csv"
predict_file <- "/Users/ethan/OneDrive/Desktop/KaggleCompetition/to_predict.csv"
products_file <- "/Users/ethan/OneDrive/Desktop/KaggleCompetition/products.csv"
output_dir <- "/Users/ethan/OneDrive/Desktop/KaggleCompetition/new_training_data"

# Load datasets
training_data <- read.csv(training_file, stringsAsFactors = FALSE)
to_predict_data <- read.csv(predict_file, stringsAsFactors = FALSE)
products_data <- read.csv(products_file, stringsAsFactors = FALSE)

# Extract valid product IDs from products.csv
valid_product_ids <- unique(products_data$product_id)

# Parse `list_product_id` into a proper list
parse_list_column <- function(column) {
  # Remove square brackets and split into a vector of strings
  str_split(gsub("\\[|\\]", "", column), "\\s+")
}

# Apply parsing function
training_data$list_product_id <- parse_list_column(training_data$list_product_id)
to_predict_data$list_product_id <- parse_list_column(to_predict_data$list_product_id)

# Extract unique customer IDs
training_customers <- unique(training_data$customer_id)
to_predict_customers <- unique(to_predict_data$customer_id)

# Analyze customer overlap
overlap_customers <- intersect(training_customers, to_predict_customers)
missing_from_to_predict <- setdiff(training_customers, to_predict_customers)

# Metrics
total_training_customers <- length(training_customers)
overlap_rate <- length(overlap_customers) / total_training_customers * 100

# Safe sampling function
safe_sample <- function(x, size, replace = FALSE) {
  if (size > length(x)) {
    warning(sprintf("Requested sample size (%d) exceeds population size (%d). Sampling all available.", size, length(x)))
    return(x)
  }
  sample(x, size, replace = replace)
}

# Generate new training sets
set.seed(42) # For reproducibility

# Proportions for 10,000 entries
num_new_customers <- floor(0.85 * 10000)
num_reused_customers <- floor(0.10 * 10000)
num_fake_customers <- floor(0.05 * 10000)

# Sample customers
selected_new_customers <- safe_sample(missing_from_to_predict, num_new_customers)
selected_reused_customers <- safe_sample(overlap_customers, num_reused_customers)

# Generate fake customer IDs
max_customer_id <- max(c(training_customers, to_predict_customers))
fake_customers <- seq(max_customer_id + 1, max_customer_id + num_fake_customers)

# Add fake customer data using only valid products
fake_customer_data <- data.frame(
  customer_id = fake_customers,
  list_product_id = replicate(num_fake_customers, paste0("[", paste(sample(valid_product_ids, 5, replace = TRUE), collapse = " "), "]"))
)

# Combine selected customers' data
new_training_data <- training_data %>%
  filter(customer_id %in% c(selected_new_customers, selected_reused_customers))

# Convert `list_product_id` to strings in `new_training_data` with brackets
new_training_data <- new_training_data %>%
  mutate(list_product_id = sapply(list_product_id, function(x) paste0("[", paste(x, collapse = " "), "]")))

# Combine with fake customer data
new_training_data <- bind_rows(new_training_data, fake_customer_data)

# Shuffle and limit to 10,000 entries
set.seed(42)
new_training_data <- new_training_data %>%
  sample_frac(1) %>%
  slice(1:10000)

# Save the first dataset without quotes
write.csv(new_training_data, file = paste0(output_dir, "/new_training_data_1.csv"), row.names = FALSE, quote = FALSE)

# Create a second shuffled dataset without quotes
set.seed(24)
new_training_data_2 <- new_training_data %>% sample_frac(1)

write.csv(new_training_data_2, file = paste0(output_dir, "/new_training_data_2.csv"), row.names = FALSE, quote = FALSE)

# Print summary information
output_summary <- list(
  total_customers = total_training_customers,
  overlapping_customers = length(overlap_customers),
  overlap_rate = round(overlap_rate, 2),
  missing_customers = length(missing_from_to_predict),
  output_files = c(
    paste0(output_dir, "/new_training_data_1.csv"),
    paste0(output_dir, "/new_training_data_2.csv")
  )
)

print(output_summary)

cat("New training datasets created:\n")
cat(paste0(output_summary$output_files, collapse = "\n"), "\n")
