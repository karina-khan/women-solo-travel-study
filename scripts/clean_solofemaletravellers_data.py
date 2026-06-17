"""
Clean r/solofemaletravellers posts dataset.
"""

import pandas as pd
import re
import numpy as np
from pathlib import Path
from datetime import datetime

# File paths
input_file = Path(__file__).parent.parent / "arctic-shift/data/r_solofemaletravellers_2018-01-01_to_2026-06-30_posts.csv"
output_file = Path(__file__).parent.parent / "data/processed/solofemaletravellers_cleaned.csv"

# Load the full dataset
print("Loading dataset...")
df = pd.read_csv(input_file)
initial_rows = len(df)

# Select only the columns we need
required_columns = ['post_id', 'title', 'selftext', 'created_utc', 'score', 'num_comments',
                    'link_flair_text', 'author', 'upvote_ratio']
df = df[required_columns].copy()

print(f"Initial dataset: {initial_rows} rows")

# Remove rows where author is [deleted] or [removed]
print("Filtering out deleted/removed posts...")
df = df[~df['author'].isin(['[deleted]', '[removed]'])]
print(f"  After author filter: {len(df)} rows ({initial_rows - len(df)} removed)")

# Remove rows where selftext is [deleted], [removed], NaN, or empty
initial_before_text = len(df)
df = df[df['selftext'].notna()]
df = df[~df['selftext'].isin(['[deleted]', '[removed]', ''])]
df = df[df['selftext'].str.strip() != '']
print(f"  After selftext filter: {len(df)} rows ({initial_before_text - len(df)} removed)")

# Count words helper function
def count_words(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

# Filter by minimum word count (title + selftext combined >= 50 words)
print("Filtering by minimum word count...")
initial_before_wordcount = len(df)
combined_wordcount = df['title'].apply(count_words) + df['selftext'].apply(count_words)
df = df[combined_wordcount >= 50].copy()
print(f"  After word count filter: {len(df)} rows ({initial_before_wordcount - len(df)} removed)")

# Clean text function
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs (http(s)://, www., etc.)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+\.com\S*', '', text)

    # Remove Reddit-specific formatting artifacts
    text = re.sub(r'&amp;#x200B;', '', text)
    text = re.sub(r'&lt;|&gt;', '', text)
    text = re.sub(r'&amp;', '&', text)

    # Convert markdown links [text](url) to just text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove excessive newlines (2+ newlines -> 1 newline)
    text = re.sub(r'\n\n+', '\n', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()

# Create full_text column
print("Creating full_text column and cleaning...")
df['full_text'] = df['title'] + ' ' + df['selftext']
df['full_text'] = df['full_text'].apply(clean_text)

# Convert created_utc to date columns
print("Converting timestamps...")
df['date'] = pd.to_datetime(df['created_utc'], unit='s')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Reorder columns for final output
final_columns = ['post_id', 'title', 'selftext', 'full_text', 'created_utc', 'date',
                 'year', 'month', 'score', 'num_comments', 'upvote_ratio',
                 'link_flair_text', 'author']
df = df[final_columns]

# Calculate statistics before saving
final_rows = len(df)
rows_removed = initial_rows - final_rows
removal_percent = (rows_removed / initial_rows) * 100

print(f"\n{'='*60}")
print(f"CLEANING SUMMARY")
print(f"{'='*60}")
print(f"Rows before: {initial_rows}")
print(f"Rows after:  {final_rows}")
print(f"Rows removed: {rows_removed} ({removal_percent:.1f}%)")

# Date range
min_date = df['date'].min()
max_date = df['date'].max()
print(f"\nDate range: {min_date.date()} to {max_date.date()}")

# Distribution of link_flair_text (top 15)
print(f"\nTop 15 link_flair_text values:")
flair_counts = df['link_flair_text'].value_counts().head(15)
for flair, count in flair_counts.items():
    pct = (count / final_rows) * 100
    print(f"  {flair or '(None)'}: {count} ({pct:.1f}%)")

# Word count statistics
full_text_wordcount = df['full_text'].apply(count_words)
avg_wordcount = full_text_wordcount.mean()
min_wordcount = full_text_wordcount.min()
max_wordcount = full_text_wordcount.max()
median_wordcount = full_text_wordcount.median()

print(f"\nFull text word count statistics:")
print(f"  Average: {avg_wordcount:.0f} words")
print(f"  Median:  {median_wordcount:.0f} words")
print(f"  Min:     {min_wordcount:.0f} words")
print(f"  Max:     {max_wordcount:.0f} words")

# Save cleaned dataset
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n✓ Cleaned dataset saved to {output_file}")
print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
