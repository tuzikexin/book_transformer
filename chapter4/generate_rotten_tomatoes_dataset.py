import pandas as pd
from datasets import load_dataset

sampling_size = 500
# Load the Rotten Tomatoes dataset from Hugging Face
dataset = load_dataset("rotten_tomatoes", split="train")

# Convert to a pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset)

# Filter positive and negative reviews
positive_reviews = df[df['label'] == 1]
negative_reviews = df[df['label'] == 0]

# Randomly select positive and negative reviews
sampled_positive_reviews = positive_reviews.sample(n=sampling_size, random_state=42)
sampled_negative_reviews = negative_reviews.sample(n=sampling_size, random_state=42)

# Combine the two samples into one DataFrame
sampled_reviews = pd.concat([sampled_positive_reviews, sampled_negative_reviews])

# Shuffle the resulting dataset to mix positive and negative reviews
sampled_reviews = sampled_reviews.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the resulting dataset to a new CSV file
sampled_reviews.to_csv('sampled_huggingface_rotten_tomatoes_reviews.csv', index=False)

print(f"Successfully sampled and saved {sampling_size} positive and {sampling_size} negative reviews.")

# Check the column names and structure of the dataset
print(sampled_reviews.head())