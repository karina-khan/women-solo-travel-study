"""
Add Reddit URLs to results folder CSVs for easy source navigation.
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("Adding URLs to Results Folder CSVs")
print("=" * 80)

results_dir = Path(__file__).parent.parent / "results"

# Function to construct Reddit URL
def construct_reddit_url(post_id, subreddit_source):
    """Construct Reddit post URL from post_id and subreddit."""
    return f"https://www.reddit.com/r/{subreddit_source}/comments/{post_id}/"

# Update growth_arc_posts_subset.csv
print("\nUpdating growth_arc_posts_subset.csv...")
df_growth = pd.read_csv(results_dir / "growth_arc_posts_subset.csv")

if 'url' not in df_growth.columns:
    df_growth['url'] = df_growth.apply(
        lambda row: construct_reddit_url(row['post_id'], row['subreddit_source']),
        axis=1
    )

    # Reorder columns to put URL after post_id
    cols = df_growth.columns.tolist()
    cols.remove('url')
    post_id_idx = cols.index('post_id')
    cols.insert(post_id_idx + 1, 'url')
    df_growth = df_growth[cols]

    df_growth.to_csv(results_dir / "growth_arc_posts_subset.csv", index=False)
    print(f"✓ Added URLs to {len(df_growth)} growth arc posts")
else:
    print("✓ URLs already present in growth_arc_posts_subset.csv")

# Update annotated_corpus_with_growth_arcs.csv
print("\nUpdating annotated_corpus_with_growth_arcs.csv...")
df_corpus = pd.read_csv(results_dir / "annotated_corpus_with_growth_arcs.csv")

if 'url' not in df_corpus.columns:
    df_corpus['url'] = df_corpus.apply(
        lambda row: construct_reddit_url(row['post_id'], row['subreddit_source']),
        axis=1
    )

    # Reorder columns to put URL after post_id
    cols = df_corpus.columns.tolist()
    cols.remove('url')
    post_id_idx = cols.index('post_id')
    cols.insert(post_id_idx + 1, 'url')
    df_corpus = df_corpus[cols]

    df_corpus.to_csv(results_dir / "annotated_corpus_with_growth_arcs.csv", index=False)
    print(f"✓ Added URLs to {len(df_corpus):,} posts in full corpus")
else:
    print("✓ URLs already present in annotated_corpus_with_growth_arcs.csv")

# Show samples
print("\n" + "=" * 80)
print("SAMPLE: Growth Arc Posts with URLs (Top 3)")
print("=" * 80)

df_growth_sample = pd.read_csv(results_dir / "growth_arc_posts_subset.csv")
for idx, row in df_growth_sample.head(3).iterrows():
    print(f"\nPost: {row['post_id']}")
    print(f"Title: {row['title'][:70]}...")
    print(f"URL: {row['url']}")
    print(f"Growth Arc Scores - Discomfort: {row['discomfort_score']:.3f}, Breakthrough: {row['breakthrough_score']:.3f}")

print("\n" + "=" * 80)
print("✅ All results CSVs updated with post URLs")
print("=" * 80)
print("\nYou can now easily navigate to the original Reddit posts!")
