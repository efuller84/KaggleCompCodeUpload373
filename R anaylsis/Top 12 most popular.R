# Load necessary libraries
library(dplyr)

# Read the transactions dataset
transactions <- read.csv("/Users/ethan/OneDrive/Desktop/KaggleCompetition/transactions.csv")

# Read the products dataset
products <- read.csv("/Users/ethan/OneDrive/Desktop/KaggleCompetition/products.csv")

# Count the number of transactions for each product
top_products <- transactions %>%
  group_by(product_id) %>%
  summarise(num_transactions = n()) %>%
  arrange(desc(num_transactions)) %>%
  slice_head(n = 12)  # Select top 12 products

# Join with the products dataset to get prod_name and prod_type_name
top_products_details <- top_products %>%
  left_join(products, by = c("product_id" = "product_id")) %>%
  select(product_id, prod_name, prod_type_name, prod_group_name, num_transactions)

# Output the table
print(top_products_details)
write.csv(top_products_details, "top_12_products.csv", row.names = FALSE)
