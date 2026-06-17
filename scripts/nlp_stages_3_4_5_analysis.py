"""
Extended NLP Analysis: Stages 3-5
- Stage 3: Internal structure of growth arc subset (topic modeling)
- Stage 4: Temporal trends (growth arc prevalence by year)
- Stage 5: Sentence-level quote extraction for video scripts
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# NLTK for sentence tokenization
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    import nltk
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic

# ============================================================================
# SETUP
# ============================================================================
print("=" * 80)
print("EXTENDED NLP ANALYSIS: Stages 3-5")
print("=" * 80)

data_dir = Path(__file__).parent.parent / "results"
analysis_dir = Path(__file__).parent.parent / "data/analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)

# Load the annotated corpus
print("\nLoading annotated corpus from Stage 1-2...")
df_full = pd.read_csv(data_dir / "annotated_corpus_with_growth_arcs.csv")
df_growth = df_full[df_full['is_growth_arc']].copy()

print(f"✓ Full corpus: {len(df_full):,} posts")
print(f"✓ Growth arc subset: {len(df_growth):,} posts")

# Load seed lists for Stage 5
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

# ============================================================================
# STAGE 3: INTERNAL STRUCTURE OF GROWTH ARC SUBSET
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 3: Topic Modeling on Growth Arc Subset")
print("=" * 80)

print("\nLoading BGE embedding model...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

print("Embedding growth arc posts...")
growth_texts = df_growth['full_text'].tolist()
growth_embeddings = model.encode(growth_texts, show_progress_bar=True, batch_size=32)
print(f"✓ Embedded {len(growth_embeddings)} posts")

print("\nRunning BERTopic on growth arc subset...")
topic_model_growth = BERTopic(
    embedding_model=model,
    nr_topics="auto",
    min_topic_size=5,
    calculate_probabilities=True,
    verbose=False
)
growth_topics, growth_probs = topic_model_growth.fit_transform(growth_texts, growth_embeddings)
df_growth['topic_id_growth'] = growth_topics

print(f"✓ Identified {len(set(growth_topics)) - 1} topics in growth arc subset")

# Extract topic info
growth_topic_info = topic_model_growth.get_topic_info()
growth_topic_info_display = growth_topic_info[growth_topic_info['Topic'] != -1].copy()
growth_topic_info_display = growth_topic_info_display.sort_values('Count', ascending=False)

print("\nTop topics in growth arc subset:")
print("-" * 80)
for idx, row in growth_topic_info_display.head(15).iterrows():
    topic_id = int(row['Topic'])
    count = int(row['Count'])
    pct = (count / len(df_growth)) * 100
    keywords = row['Representation'][:5] if isinstance(row['Representation'], list) else []
    keywords_str = ", ".join(keywords) if keywords else "N/A"
    print(f"Topic {topic_id:2d} | {count:4d} posts ({pct:5.1f}%) | {keywords_str}")

# Save Stage 3 results
growth_topic_summary_rows = []
for idx, row in growth_topic_info[growth_topic_info['Topic'] != -1].iterrows():
    topic_id = int(row['Topic'])
    keywords = row['Representation'][:10] if isinstance(row['Representation'], list) else []
    keywords_str = " | ".join(keywords)
    growth_topic_summary_rows.append({
        'Topic': topic_id,
        'Size': int(row['Count']),
        'Top_10_Keywords': keywords_str
    })

growth_topic_df = pd.DataFrame(growth_topic_summary_rows)
growth_topic_df = growth_topic_df.sort_values('Size', ascending=False)
growth_topic_df.to_csv(analysis_dir / "stage3_growth_arc_topics.csv", index=False)
print(f"\n✓ Saved: {analysis_dir / 'stage3_growth_arc_topics.csv'}")

# ============================================================================
# STAGE 4: TEMPORAL TRENDS
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 4: Temporal Trends of Growth Arc Language")
print("=" * 80)

# Convert date to datetime for grouping
df_full['date'] = pd.to_datetime(df_full['date'])

# Group by year
temporal_data = []
for year in sorted(df_full['year'].unique()):
    year_data = df_full[df_full['year'] == year]
    total_posts = len(year_data)
    growth_arc_posts = year_data['is_growth_arc'].sum()
    growth_arc_pct = (growth_arc_posts / total_posts * 100) if total_posts > 0 else 0

    temporal_data.append({
        'Year': int(year),
        'Total_Posts': total_posts,
        'Growth_Arc_Posts': int(growth_arc_posts),
        'Growth_Arc_Percentage': round(growth_arc_pct, 1)
    })

temporal_df = pd.DataFrame(temporal_data)
temporal_df.to_csv(analysis_dir / "stage4_temporal_trends.csv", index=False)

print("\nGrowth Arc Prevalence by Year:")
print("-" * 80)
print(temporal_df.to_string(index=False))

# Generate matplotlib chart
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(temporal_df['Year'], temporal_df['Growth_Arc_Percentage'],
        marker='o', linewidth=2.5, markersize=8, color='#E74C3C')
ax.fill_between(temporal_df['Year'], temporal_df['Growth_Arc_Percentage'],
                alpha=0.3, color='#E74C3C')

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Growth Arc Posts (%)', fontsize=12, fontweight='bold')
ax.set_title('Growth Arc Language Prevalence Over Time\n(Posts containing both discomfort + breakthrough language)',
             fontsize=13, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 25)

# Add value labels on points
for year, pct in zip(temporal_df['Year'], temporal_df['Growth_Arc_Percentage']):
    ax.annotate(f'{pct:.1f}%', xy=(year, pct), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
chart_path = analysis_dir / "stage4_growth_arc_trend.png"
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {chart_path}")
plt.close()

print(f"\n✓ Saved: {analysis_dir / 'stage4_temporal_trends.csv'}")

# ============================================================================
# STAGE 5: SENTENCE-LEVEL QUOTE EXTRACTION
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 5: Sentence-Level Quote Extraction")
print("=" * 80)

print("\nEmbedding seed lists...")
discomfort_embeddings = model.encode(discomfort_seeds, show_progress_bar=False)
breakthrough_embeddings = model.encode(breakthrough_seeds, show_progress_bar=False)

print("Extracting and scoring sentences...")
quotes_data = []
sentence_count = 0

for idx, row in df_growth.iterrows():
    if idx % 200 == 0:
        print(f"  Processing post {idx + 1}/{len(df_growth)}...")

    post_id = row['post_id']
    year = row['year']
    subreddit = row['subreddit_source']
    full_text = row['full_text']

    # Split into sentences
    try:
        sentences = sent_tokenize(full_text)
    except:
        sentences = full_text.split('. ')

    if not sentences:
        continue

    # Embed sentences
    sentence_embeddings = model.encode(sentences, show_progress_bar=False)

    # Score against both seed lists
    discomfort_sims = cosine_similarity(sentence_embeddings, discomfort_embeddings)
    breakthrough_sims = cosine_similarity(sentence_embeddings, breakthrough_embeddings)

    for sent_idx, sentence in enumerate(sentences):
        if len(sentence.strip()) < 20:  # Skip very short sentences
            continue

        discomfort_score = discomfort_sims[sent_idx].max()
        breakthrough_score = breakthrough_sims[sent_idx].max()

        # Add both if above threshold
        if discomfort_score > 0.5:
            quotes_data.append({
                'post_id': post_id,
                'sentence': sentence.strip(),
                'score': round(float(discomfort_score), 3),
                'type': 'discomfort',
                'year': int(year),
                'subreddit_source': subreddit
            })

        if breakthrough_score > 0.5:
            quotes_data.append({
                'post_id': post_id,
                'sentence': sentence.strip(),
                'score': round(float(breakthrough_score), 3),
                'type': 'breakthrough',
                'year': int(year),
                'subreddit_source': subreddit
            })

        sentence_count += 1

print(f"✓ Extracted {sentence_count} sentences")

# Convert to DataFrame and sort
quotes_df = pd.DataFrame(quotes_data)

# Get top 50 of each type
discomfort_quotes = quotes_df[quotes_df['type'] == 'discomfort'].nlargest(50, 'score')
breakthrough_quotes = quotes_df[quotes_df['type'] == 'breakthrough'].nlargest(50, 'score')

# Save separately
discomfort_quotes.to_csv(analysis_dir / "stage5_discomfort_quotes.csv", index=False)
breakthrough_quotes.to_csv(analysis_dir / "stage5_breakthrough_quotes.csv", index=False)

print(f"\n✓ Saved: {analysis_dir / 'stage5_discomfort_quotes.csv'} ({len(discomfort_quotes)} quotes)")
print(f"✓ Saved: {analysis_dir / 'stage5_breakthrough_quotes.csv'} ({len(breakthrough_quotes)} quotes)")

# Also save combined for easy browsing
all_quotes = pd.concat([discomfort_quotes, breakthrough_quotes]).sort_values('score', ascending=False)
all_quotes.to_csv(analysis_dir / "stage5_all_top_quotes.csv", index=False)
print(f"✓ Saved: {analysis_dir / 'stage5_all_top_quotes.csv'} ({len(all_quotes)} quotes)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE: All Stages Done")
print("=" * 80)

print(f"\nResults directory: {analysis_dir}")
print("\nGenerated files:")
print("  STAGE 3:")
print("    - stage3_growth_arc_topics.csv")
print("  STAGE 4:")
print("    - stage4_temporal_trends.csv")
print("    - stage4_growth_arc_trend.png")
print("  STAGE 5:")
print("    - stage5_discomfort_quotes.csv (top 50)")
print("    - stage5_breakthrough_quotes.csv (top 50)")
print("    - stage5_all_top_quotes.csv (combined, sorted by score)")

print("\n" + "=" * 80)
print("INSIGHTS SUMMARY")
print("=" * 80)
print(f"\nStage 3: Growth arc posts cluster into {len(set(growth_topics)) - 1} distinct topics")
print(f"         Top themes: {', '.join(growth_topic_info_display.iloc[0]['Representation'][:3])}")

print(f"\nStage 4: Growth arc prevalence trend")
print(f"         {temporal_df['Year'].min()}: {temporal_df.iloc[0]['Growth_Arc_Percentage']:.1f}%")
print(f"         {temporal_df['Year'].max()}: {temporal_df.iloc[-1]['Growth_Arc_Percentage']:.1f}%")
trend = temporal_df.iloc[-1]['Growth_Arc_Percentage'] - temporal_df.iloc[0]['Growth_Arc_Percentage']
direction = "↑ INCREASING" if trend > 0 else "↓ DECREASING"
print(f"         Overall change: {direction} ({trend:+.1f}%)")

print(f"\nStage 5: High-quality quotes for video scripts")
print(f"         Top discomfort quote score: {discomfort_quotes.iloc[0]['score']:.3f}")
print(f"         Top breakthrough quote score: {breakthrough_quotes.iloc[0]['score']:.3f}")

print("\n✅ Ready for: narrative research, video script generation, qualitative analysis")
