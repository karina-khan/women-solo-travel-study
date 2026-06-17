"""
NLP Analysis Pipeline: Topic Modeling + Personal Growth Semantic Analysis

1. Merges two Reddit datasets (femaletravels, solofemaletravellers)
2. Performs exploratory topic modeling with BERTopic
3. Semantic zoom analysis on personal growth narratives
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check and install required packages
print("Checking dependencies...")
required_packages = {
    'sentence_transformers': 'sentence-transformers',
    'bertopic': 'bertopic',
    'umap': 'umap-learn',
    'hdbscan': 'hdbscan'
}

missing_packages = []
for module, pip_name in required_packages.items():
    try:
        __import__(module)
    except ImportError:
        missing_packages.append(pip_name)

if missing_packages:
    print(f"Installing missing packages: {', '.join(missing_packages)}")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q'] + missing_packages)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
from bertopic import BERTopic

print("✓ All dependencies ready\n")

# ============================================================================
# SETUP: Load and merge datasets
# ============================================================================
print("=" * 80)
print("SETUP: Loading and merging datasets")
print("=" * 80)

data_dir = Path(__file__).parent.parent / "data/processed"

df_solo = pd.read_csv(data_dir / "solofemaletravellers_cleaned.csv")
df_female = pd.read_csv(data_dir / "femaletravels_cleaned.csv")

df_solo['subreddit_source'] = 'solofemaletravellers'
df_female['subreddit_source'] = 'femaletravels'

# Merge
df = pd.concat([df_solo, df_female], ignore_index=True)

print(f"Loaded r/solofemaletravellers: {len(df_solo):,} posts")
print(f"Loaded r/femaletravels: {len(df_female):,} posts")
print(f"Combined corpus: {len(df):,} posts")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Remove any NaN in full_text
df = df[df['full_text'].notna()].copy()
df = df[df['full_text'].str.strip() != '']
print(f"After removing empty texts: {len(df):,} posts\n")

# ============================================================================
# STAGE 1: Exploratory Topic Modeling
# ============================================================================
print("=" * 80)
print("STAGE 1: Exploratory Topic Modeling")
print("=" * 80)

print("\nLoading BGE embedding model (BAAI/bge-small-en-v1.5)...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

print("Embedding full_text corpus...")
texts = df['full_text'].tolist()
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
print(f"✓ Embedded {len(embeddings)} posts (dimension: {embeddings.shape[1]})")

print("\nRunning BERTopic...")
topic_model = BERTopic(
    embedding_model=model,
    nr_topics="auto",
    min_topic_size=10,
    calculate_probabilities=True,
    verbose=False
)
topics, probs = topic_model.fit_transform(texts, embeddings)
df['topic_id'] = topics

print(f"✓ Identified {len(set(topics)) - 1} topics (excluding noise)")

# Extract topic information
topic_info = topic_model.get_topic_info()
topic_info_display = topic_info[topic_info['Topic'] != -1].copy()
topic_info_display = topic_info_display.sort_values('Count', ascending=False)

print("\nTop 20 Topics by Size:")
print("-" * 80)
for idx, row in topic_info_display.head(20).iterrows():
    topic_id = int(row['Topic'])
    count = int(row['Count'])
    pct = (count / len(df)) * 100
    keywords = row['Representation'][:5] if isinstance(row['Representation'], list) else []
    keywords_str = ", ".join(keywords) if keywords else "N/A"
    print(f"Topic {topic_id:3d} | {count:5d} posts ({pct:5.1f}%) | {keywords_str}")

# Save topic info to CSV
output_dir = Path(__file__).parent.parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

# Detailed topic CSV with top 10 keywords
topic_summary_rows = []
for idx, row in topic_info[topic_info['Topic'] != -1].iterrows():
    topic_id = int(row['Topic'])
    keywords = row['Representation'][:10] if isinstance(row['Representation'], list) else []
    keywords_str = " | ".join(keywords)
    topic_summary_rows.append({
        'Topic': topic_id,
        'Size': int(row['Count']),
        'Top_10_Keywords': keywords_str
    })

topic_summary_df = pd.DataFrame(topic_summary_rows)
topic_summary_df = topic_summary_df.sort_values('Size', ascending=False)
topic_summary_df.to_csv(output_dir / "topics_summary.csv", index=False)
print(f"\n✓ Saved: {output_dir / 'topics_summary.csv'}")

# Representative posts per topic
print("\nExtracting most representative posts per topic...")
representative_posts_rows = []
for topic_id in sorted(set(topics)):
    if topic_id == -1:  # Skip noise cluster
        continue
    topic_mask = df['topic_id'] == topic_id
    topic_texts = df[topic_mask].copy()

    # Get the most representative documents from BERTopic
    representative_docs = topic_model.get_representative_docs(topic_id)

    for i, doc in enumerate(representative_docs[:5]):
        # Find the post that matches this document
        matches = topic_texts[topic_texts['full_text'] == doc]
        if len(matches) > 0:
            match = matches.iloc[0]
            representative_posts_rows.append({
                'Topic': topic_id,
                'Rank': i + 1,
                'Post_ID': match['post_id'],
                'Subreddit': match['subreddit_source'],
                'Date': match['date'],
                'Score': match['score'],
                'Num_Comments': match['num_comments'],
                'Title': match['title'][:100] + '...' if len(match['title']) > 100 else match['title'],
                'Text_Preview': doc[:200] + '...' if len(doc) > 200 else doc
            })

representative_posts_df = pd.DataFrame(representative_posts_rows)
representative_posts_df.to_csv(output_dir / "representative_posts_per_topic.csv", index=False)
print(f"✓ Saved: {output_dir / 'representative_posts_per_topic.csv'}")

# Save intertopic distance map
print("\nGenerating intertopic distance visualization...")
try:
    fig = topic_model.visualize_topics()
    fig.write_html(output_dir / "intertopic_distance_map.html")
    print(f"✓ Saved: {output_dir / 'intertopic_distance_map.html'}")
except Exception as e:
    print(f"⚠ Could not generate intertopic visualization: {e}")

print("\n" + "=" * 80)
print("STAGE 1 COMPLETE")
print("=" * 80)

# ============================================================================
# STAGE 2: Semantic Zoom on Personal Growth
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 2: Semantic Zoom on Personal Growth")
print("=" * 80)

# Define seed sentences
discomfort_seeds = [
    "I'm enjoying myself but I also feel so alone",
    "My anxiety is making it hard to just go out and explore on my own",
    "This is completely outside my comfort zone and I'm not sure I can do it",
    "I'm scared something bad is going to happen to me while I'm alone",
    "I had a good day but I still feel lonely sometimes",
    "I can't get the courage to do this by myself because of my anxiety"
]

breakthrough_seeds = [
    "This turned out to be one of the most life changing experiences I've had",
    "I realized I don't need to wait for someone else to do the things I want",
    "I feel this anticipation and excitement about stepping into the unknown",
    "I found the courage to do something I never thought I could do alone",
    "Doing this by myself taught me to trust my own judgment",
    "I learned something about myself that I didn't expect"
]

print(f"\nDiscomfort seeds: {len(discomfort_seeds)}")
for i, seed in enumerate(discomfort_seeds, 1):
    print(f"  {i}. {seed}")

print(f"\nBreakthrough seeds: {len(breakthrough_seeds)}")
for i, seed in enumerate(breakthrough_seeds, 1):
    print(f"  {i}. {seed}")

# Embed seed sentences
print("\nEmbedding seed sentences...")
discomfort_embeddings = model.encode(discomfort_seeds, show_progress_bar=False)
breakthrough_embeddings = model.encode(breakthrough_seeds, show_progress_bar=False)
print(f"✓ Embedded discomfort seeds: {discomfort_embeddings.shape}")
print(f"✓ Embedded breakthrough seeds: {breakthrough_embeddings.shape}")

# Compute similarity scores
print("\nComputing semantic similarity scores...")
discomfort_similarities = cosine_similarity(embeddings, discomfort_embeddings)
breakthrough_similarities = cosine_similarity(embeddings, breakthrough_embeddings)

df['discomfort_score'] = discomfort_similarities.max(axis=1)
df['breakthrough_score'] = breakthrough_similarities.max(axis=1)

print(f"✓ Discomfort scores - mean: {df['discomfort_score'].mean():.3f}, std: {df['discomfort_score'].std():.3f}")
print(f"✓ Breakthrough scores - mean: {df['breakthrough_score'].mean():.3f}, std: {df['breakthrough_score'].std():.3f}")

# Compute thresholds (75th percentile)
discomfort_threshold = df['discomfort_score'].quantile(0.75)
breakthrough_threshold = df['breakthrough_score'].quantile(0.75)

print(f"\nThresholds (75th percentile):")
print(f"  Discomfort threshold: {discomfort_threshold:.3f}")
print(f"  Breakthrough threshold: {breakthrough_threshold:.3f}")

# Create boolean columns
df['is_discomfort'] = df['discomfort_score'] >= discomfort_threshold
df['is_breakthrough'] = df['breakthrough_score'] >= breakthrough_threshold
df['is_growth_arc'] = df['is_discomfort'] & df['is_breakthrough']

# Summary statistics
total_posts = len(df)
discomfort_count = df['is_discomfort'].sum()
breakthrough_count = df['is_breakthrough'].sum()
growth_arc_count = df['is_growth_arc'].sum()

print("\n" + "=" * 80)
print("SEMANTIC CATEGORY SUMMARY")
print("=" * 80)
print(f"\nTotal posts: {total_posts:,}")
print(f"\nDiscomfort language:")
print(f"  Posts: {discomfort_count:,} ({100 * discomfort_count / total_posts:.1f}%)")
print(f"\nBreakthrough language:")
print(f"  Posts: {breakthrough_count:,} ({100 * breakthrough_count / total_posts:.1f}%)")
print(f"\nGrowth Arc (both discomfort + breakthrough):")
print(f"  Posts: {growth_arc_count:,} ({100 * growth_arc_count / total_posts:.1f}%)")

# Breakdown by subreddit
print("\n" + "-" * 80)
print("Breakdown by subreddit source:")
print("-" * 80)

for subreddit in df['subreddit_source'].unique():
    sub_df = df[df['subreddit_source'] == subreddit]
    sub_total = len(sub_df)
    sub_discomfort = sub_df['is_discomfort'].sum()
    sub_breakthrough = sub_df['is_breakthrough'].sum()
    sub_growth = sub_df['is_growth_arc'].sum()

    print(f"\nr/{subreddit} ({sub_total:,} posts):")
    print(f"  Discomfort: {sub_discomfort:,} ({100 * sub_discomfort / sub_total:.1f}%)")
    print(f"  Breakthrough: {sub_breakthrough:,} ({100 * sub_breakthrough / sub_total:.1f}%)")
    print(f"  Growth Arc: {sub_growth:,} ({100 * sub_growth / sub_total:.1f}%)")

# Save annotated dataset
print("\n" + "-" * 80)
print("Saving annotated corpus...")

output_columns = ['post_id', 'subreddit_source', 'date', 'year', 'title', 'full_text',
                  'score', 'num_comments', 'topic_id', 'discomfort_score',
                  'breakthrough_score', 'is_discomfort', 'is_breakthrough', 'is_growth_arc']
df_output = df[output_columns].copy()
df_output = df_output.sort_values('is_growth_arc', ascending=False)
df_output.to_csv(output_dir / "annotated_corpus_with_growth_arcs.csv", index=False)
print(f"✓ Saved: {output_dir / 'annotated_corpus_with_growth_arcs.csv'}")

# Save growth arc posts separately (high-quality subset)
growth_arc_posts = df[df['is_growth_arc']].copy()
growth_arc_posts = growth_arc_posts.sort_values('discomfort_score', ascending=False)
growth_arc_posts[output_columns].to_csv(output_dir / "growth_arc_posts_subset.csv", index=False)
print(f"✓ Saved: {output_dir / 'growth_arc_posts_subset.csv'} ({len(growth_arc_posts)} posts)")

print("\n" + "=" * 80)
print("STAGE 2 COMPLETE - Analysis finished!")
print("=" * 80)
print(f"\nResults saved to: {output_dir}")
print("\nGenerated files:")
print("  - topics_summary.csv")
print("  - representative_posts_per_topic.csv")
print("  - intertopic_distance_map.html")
print("  - annotated_corpus_with_growth_arcs.csv")
print("  - growth_arc_posts_subset.csv")
