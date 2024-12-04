# Load necessary library
library(dplyr)

# Read the dataset
products_file <- "/Users/ethan/OneDrive/Desktop/KaggleCompetition/products.csv"
products <- read.csv(products_file)

# Count occurrences of each prod_group_name
prod_group_counts <- products %>%
  group_by(prod_group_name) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Print the result
print(prod_group_counts)

# Split the data into Garment and Non-Garment sections
garment_data <- products %>%
  filter(grepl("Garment", prod_group_name))  # Filter rows with 'Garment' in prod_group_name

non_garment_data <- products %>%
  filter(!grepl("Garment", prod_group_name))  # Filter rows without 'Garment' in prod_group_name

# View a summary of the split datasets
print(paste("Number of Garment items:", nrow(garment_data)))
print(paste("Number of Non-Garment items:", nrow(non_garment_data)))

# Save the split datasets for further analysis (optional)
write.csv(garment_data, "/Users/ethan/OneDrive/Desktop/KaggleCompetition/garment_data.csv", row.names = FALSE)
write.csv(non_garment_data, "/Users/ethan/OneDrive/Desktop/KaggleCompetition/non_garment_data.csv", row.names = FALSE)
