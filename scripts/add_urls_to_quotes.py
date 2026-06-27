"""
Add Reddit URLs to stage 5 quote CSVs for easy source navigation.
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("Adding URLs to Stage 5 Quote CSVs")
print("=" * 80)

results_dir = Path(__file__).parent.parent / "results"
analysis_dir = Path(__file__).parent.parent / "data/analysis"

# Load the annotated corpus to get URL information
print("\nLoading annotated corpus...")
df_corpus = pd.read_csv(results_dir / "annotated_corpus_with_growth_arcs.csv")
print(f"✓ Loaded {len(df_corpus):,} posts")

# Create a mapping of post_id -> subreddit_source
post_to_subreddit = dict(zip(df_corpus['post_id'], df_corpus['subreddit_source']))

# Function to construct Reddit URL
def construct_reddit_url(post_id, subreddit_source):
    """Construct Reddit post URL from post_id and subreddit."""
    return f"https://www.reddit.com/r/{subreddit_source}/comments/{post_id}/"

# Update discomfort quotes
print("\nUpdating stage5_discomfort_quotes.csv...")
df_discomfort = pd.read_csv(analysis_dir / "stage5_discomfort_quotes.csv")
df_discomfort['url'] = df_discomfort.apply(
    lambda row: construct_reddit_url(row['post_id'], row['subreddit_source']),
    axis=1
)

# Reorder columns to put URL after post_id
cols = df_discomfort.columns.tolist()
cols.remove('url')
post_id_idx = cols.index('post_id')
cols.insert(post_id_idx + 1, 'url')
df_discomfort = df_discomfort[cols]

df_discomfort.to_csv(analysis_dir / "stage5_discomfort_quotes.csv", index=False)
print(f"✓ Updated {len(df_discomfort)} discomfort quotes with URLs")

# Update breakthrough quotes
print("Updating stage5_breakthrough_quotes.csv...")
df_breakthrough = pd.read_csv(analysis_dir / "stage5_breakthrough_quotes.csv")
df_breakthrough['url'] = df_breakthrough.apply(
    lambda row: construct_reddit_url(row['post_id'], row['subreddit_source']),
    axis=1
)

# Reorder columns
cols = df_breakthrough.columns.tolist()
cols.remove('url')
post_id_idx = cols.index('post_id')
cols.insert(post_id_idx + 1, 'url')
df_breakthrough = df_breakthrough[cols]

df_breakthrough.to_csv(analysis_dir / "stage5_breakthrough_quotes.csv", index=False)
print(f"✓ Updated {len(df_breakthrough)} breakthrough quotes with URLs")

# Update combined quotes
print("Updating stage5_all_top_quotes.csv...")
df_all = pd.read_csv(analysis_dir / "stage5_all_top_quotes.csv")
df_all['url'] = df_all.apply(
    lambda row: construct_reddit_url(row['post_id'], row['subreddit_source']),
    axis=1
)

# Reorder columns
cols = df_all.columns.tolist()
cols.remove('url')
post_id_idx = cols.index('post_id')
cols.insert(post_id_idx + 1, 'url')
df_all = df_all[cols]

df_all.to_csv(analysis_dir / "stage5_all_top_quotes.csv", index=False)
print(f"✓ Updated {len(df_all)} combined quotes with URLs")

# Show sample
print("\n" + "=" * 80)
print("SAMPLE: Top 3 Discomfort Quotes with URLs")
print("=" * 80)
for idx, row in df_discomfort.head(3).iterrows():
    print(f"\n[{row['type'].upper()}] Score: {row['score']}")
    print(f"Sentence: {row['sentence'][:80]}...")
    print(f"Source: {row['url']}")

print("\n" + "=" * 80)
print("✅ All quote CSVs updated with post URLs")
print("=" * 80)
print("\nYou can now click/paste the URLs to view the original Reddit posts!")
