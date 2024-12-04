# Load necessary libraries
library(dplyr)
library(ggplot2)

# Read the datasets
products_file <- "/Users/ethan/OneDrive/Desktop/KaggleCompetition/products.csv"
transactions_file <- "/Users/ethan/OneDrive/Desktop/KaggleCompetition/transactions.csv"

products <- read.csv(products_file)
transactions <- read.csv(transactions_file)

# Merge transactions with product data to associate transactions with prod_group_name
transaction_details <- transactions %>%
  left_join(products, by = "product_id")

# Filter for the top 9 product groups
top_9_prod_groups <- c(
  "Garment Upper body", "Garment Lower body", "Garment Full body",
  "Accessories", "Underwear", "Shoes", 
  "Swimwear", "Socks & Tights", "Nightwear"
)

top_9_transactions <- transaction_details %>%
  filter(prod_group_name %in% top_9_prod_groups) %>%
  group_by(prod_group_name) %>%
  summarise(transaction_count = n()) %>%
  arrange(desc(transaction_count))

# Plot the bar graph
ggplot(top_9_transactions, aes(x = reorder(prod_group_name, -transaction_count), y = transaction_count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    title = "Popularity of Top 9 Product Group Names by Transactions",
    x = "Product Group Name",
    y = "Total Transaction Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save the plot (optional)
ggsave("/Users/ethan/OneDrive/Desktop/KaggleCompetition/top_9_prod_group_bargraph.png")
