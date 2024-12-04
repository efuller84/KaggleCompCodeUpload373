# Load necessary libraries
library(ggplot2)
library(lubridate)
library(dplyr)
transactions <- training_file <- "/Users/ethan/OneDrive/Desktop/KaggleCompetition/transactions.csv"
# Ensure date is in Date format
transactions$time_date <- as.Date(transactions$time_date)

# Extract month and year, and create the combined month-year column
transactions$month_year <- paste(year(transactions$time_date), month(transactions$time_date), sep = "-")

# Summarize average price per month-year
avg_price_by_month_year <- transactions %>%
  group_by(month_year) %>%
  summarize(avg_price = mean(price, na.rm = TRUE))

# Convert month-year to a date format (first day of each month)
avg_price_by_month_year$month_year <- as.Date(paste0(avg_price_by_month_year$month_year, "-01"))

# Plot with ggplot2
ggplot(avg_price_by_month_year, aes(x = month_year, y = avg_price)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Average Price by Month-Year", x = "Month-Year", y = "Average Price") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels for readability

