"""
Temporal Analysis by Subreddit: Growth Arc Language Trends (2018-2026)

Compares growth arc prevalence between r/solofemaletravellers and r/femaletravels
with stratified temporal analysis and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 90)
print("TEMPORAL ANALYSIS BY SUBREDDIT: Growth Arc Language Trends")
print("=" * 90)

# Load data
results_dir = Path(__file__).parent.parent / "results"
analysis_dir = Path(__file__).parent.parent / "data/analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(results_dir / "annotated_corpus_with_growth_arcs.csv")
print(f"\n✓ Loaded {len(df):,} posts from annotated corpus")

# Convert boolean columns
df['is_growth_arc'] = df['is_growth_arc'].astype(bool)

# ============================================================================
# GROUP BY YEAR AND SUBREDDIT
# ============================================================================
print("\nCalculating year/subreddit statistics...")

yearly_subreddit = []

for year in sorted(df['year'].unique()):
    year_data = df[df['year'] == year]

    for subreddit in sorted(year_data['subreddit_source'].unique()):
        sub_data = year_data[year_data['subreddit_source'] == subreddit]

        total_posts = len(sub_data)
        growth_arc_count = sub_data['is_growth_arc'].sum()
        growth_arc_pct = (growth_arc_count / total_posts * 100) if total_posts > 0 else 0
        low_sample_size = total_posts < 30

        yearly_subreddit.append({
            'year': int(year),
            'subreddit_source': subreddit,
            'total_posts': total_posts,
            'growth_arc_count': int(growth_arc_count),
            'growth_arc_pct': round(growth_arc_pct, 1),
            'low_sample_size': low_sample_size
        })

# Convert to DataFrame
results_df = pd.DataFrame(yearly_subreddit)
results_df.to_csv(analysis_dir / "stage4_temporal_trends_by_subreddit.csv", index=False)
print(f"\n✓ Saved: {analysis_dir / 'stage4_temporal_trends_by_subreddit.csv'}")

# ============================================================================
# SUMMARY TABLE: Posts by Year and Subreddit
# ============================================================================
print("\n" + "=" * 90)
print("SUMMARY: Post Distribution by Year and Subreddit")
print("=" * 90)

summary_pivot = results_df.pivot_table(
    index='year',
    columns='subreddit_source',
    values='total_posts',
    aggfunc='sum'
).fillna(0).astype(int)

summary_pivot['TOTAL'] = summary_pivot.sum(axis=1)

print("\n" + summary_pivot.to_string())
print(f"\nTotal posts: {summary_pivot['TOTAL'].sum():,}")
print(f"  r/femaletravels: {summary_pivot.get('femaletravels', 0).sum():,} ({summary_pivot.get('femaletravels', 0).sum() / summary_pivot['TOTAL'].sum() * 100:.1f}%)")
print(f"  r/solofemaletravellers: {summary_pivot.get('solofemaletravellers', 0).sum():,} ({summary_pivot.get('solofemaletravellers', 0).sum() / summary_pivot['TOTAL'].sum() * 100:.1f}%)")

# ============================================================================
# VISUALIZATION: Two Subreddits + Combined Trend
# ============================================================================
print("\n" + "=" * 90)
print("Generating visualization...")
print("=" * 90)

fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for each subreddit
solo = results_df[results_df['subreddit_source'] == 'solofemaletravellers'].sort_values('year')
female = results_df[results_df['subreddit_source'] == 'femaletravels'].sort_values('year')

# Calculate combined trend (re-compute from original data)
combined_trend = []
for year in sorted(df['year'].unique()):
    year_data = df[df['year'] == year]
    total = len(year_data)
    growth_count = year_data['is_growth_arc'].sum()
    pct = (growth_count / total * 100) if total > 0 else 0
    combined_trend.append({'year': year, 'growth_arc_pct': pct})

combined_df = pd.DataFrame(combined_trend).sort_values('year')

# Plot combined trend first (background, lighter)
ax.plot(combined_df['year'], combined_df['growth_arc_pct'],
        label='Combined (both subreddits)', color='#cccccc', linestyle='--',
        linewidth=2, marker='o', markersize=5, zorder=1)

# Plot r/solofemaletravellers
filled_solo = solo[~solo['low_sample_size']]
hollow_solo = solo[solo['low_sample_size']]

ax.plot(filled_solo['year'], filled_solo['growth_arc_pct'],
        label='r/solofemaletravellers', color='#2E86AB', linestyle='-',
        linewidth=2.5, marker='o', markersize=7, zorder=2)

# Hollow markers for low sample size
ax.scatter(hollow_solo['year'], hollow_solo['growth_arc_pct'],
           color='#2E86AB', marker='o', s=100, facecolors='none',
           edgecolors='#2E86AB', linewidth=2.5, zorder=2)

# Add text labels for low-sample points (solofemaletravellers)
for _, row in hollow_solo.iterrows():
    ax.annotate(f"n={row['total_posts']}",
                xy=(row['year'], row['growth_arc_pct']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, color='#2E86AB', weight='bold')

# Plot r/femaletravels
filled_female = female[~female['low_sample_size']]
hollow_female = female[female['low_sample_size']]

ax.plot(filled_female['year'], filled_female['growth_arc_pct'],
        label='r/femaletravels', color='#A23B72', linestyle='-',
        linewidth=2.5, marker='s', markersize=7, zorder=2)

# Hollow markers for low sample size
ax.scatter(hollow_female['year'], hollow_female['growth_arc_pct'],
           color='#A23B72', marker='s', s=100, facecolors='none',
           edgecolors='#A23B72', linewidth=2.5, zorder=2)

# Add text labels for low-sample points (femaletravels)
for _, row in hollow_female.iterrows():
    ax.annotate(f"n={row['total_posts']}",
                xy=(row['year'], row['growth_arc_pct']),
                xytext=(5, -12), textcoords='offset points',
                fontsize=8, color='#A23B72', weight='bold')

# Formatting
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Growth Arc Posts (%)', fontsize=12, fontweight='bold')
ax.set_title('Growth Arc Language Prevalence Over Time\nStratified by Subreddit (2018-2026)\n' +
             'Hollow markers indicate low sample size (n < 30)',
             fontsize=13, fontweight='bold', pad=20)

ax.set_xlim(2017.5, 2026.5)
ax.set_ylim(0, max(
    solo['growth_arc_pct'].max(),
    female['growth_arc_pct'].max(),
    combined_df['growth_arc_pct'].max()
) + 5)

ax.grid(True, alpha=0.3, linestyle=':')
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

# Add note about sample sizes
note_text = "Note: Hollow markers indicate years with < 30 posts for that subreddit (statistically unreliable)"
fig.text(0.5, 0.01, note_text, ha='center', fontsize=9, style='italic', color='#666666')

plt.tight_layout(rect=[0, 0.02, 1, 1])
chart_path = analysis_dir / "stage4_temporal_trends_by_subreddit.png"
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {chart_path}")
plt.close()

# ============================================================================
# DETAILED STATISTICS BY SUBREDDIT
# ============================================================================
print("\n" + "=" * 90)
print("DETAILED STATISTICS BY SUBREDDIT")
print("=" * 90)

for subreddit in sorted(results_df['subreddit_source'].unique()):
    sub_df = results_df[results_df['subreddit_source'] == subreddit]
    print(f"\nr/{subreddit}:")
    print("-" * 90)

    # Overall stats
    total_posts = sub_df['total_posts'].sum()
    total_growth = sub_df['growth_arc_count'].sum()
    overall_pct = (total_growth / total_posts * 100) if total_posts > 0 else 0

    print(f"  Total posts: {total_posts:,}")
    print(f"  Total growth arc posts: {total_growth:,}")
    print(f"  Overall growth arc percentage: {overall_pct:.1f}%")
    print(f"  Years with data: {len(sub_df)}")
    print(f"  Low sample size years: {sub_df['low_sample_size'].sum()}")

    # Year-by-year
    print(f"\n  Year-by-year breakdown:")
    print(f"  {'Year':<6} {'Posts':<8} {'Growth':<8} {'%':<6} {'Flag':<6}")
    print(f"  {'-' * 34}")

    for _, row in sub_df.sort_values('year').iterrows():
        flag = "⚠️ LOW" if row['low_sample_size'] else ""
        print(f"  {int(row['year']):<6} {row['total_posts']:<8} {row['growth_arc_count']:<8} " +
              f"{row['growth_arc_pct']:<6.1f} {flag:<6}")

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 90)
print("KEY INSIGHTS")
print("=" * 90)

solo_2024 = results_df[(results_df['year'] == 2024) & (results_df['subreddit_source'] == 'solofemaletravellers')]
female_2024 = results_df[(results_df['year'] == 2024) & (results_df['subreddit_source'] == 'femaletravels')]

if not solo_2024.empty and not female_2024.empty:
    solo_pct = solo_2024.iloc[0]['growth_arc_pct']
    female_pct = female_2024.iloc[0]['growth_arc_pct']
    print(f"\n1. SUBREDDIT COMPARISON (2024):")
    print(f"   r/solofemaletravellers: {solo_pct:.1f}% growth arc language")
    print(f"   r/femaletravels: {female_pct:.1f}% growth arc language")
    print(f"   → r/solofemaletravellers is {abs(solo_pct - female_pct):.1f} percentage points {'higher' if solo_pct > female_pct else 'lower'}")

print(f"\n2. TEMPORAL SHIFT:")
early = results_df[results_df['year'].isin([2018, 2019, 2020])]['growth_arc_pct'].mean()
recent = results_df[results_df['year'].isin([2024, 2025, 2026])]['growth_arc_pct'].mean()
print(f"   2018-2020 average: {early:.1f}%")
print(f"   2024-2026 average: {recent:.1f}%")
print(f"   → Trend: {'+' if recent > early else '-'}{abs(recent - early):.1f} percentage points")

low_sample = results_df[results_df['low_sample_size']]
print(f"\n3. SAMPLE SIZE WARNINGS:")
print(f"   {len(low_sample)} year/subreddit combinations have < 30 posts")
print(f"   These are marked with hollow markers and n-counts on the chart")
print(f"   → Use caution when interpreting these data points")

print("\n" + "=" * 90)
print("✅ ANALYSIS COMPLETE")
print("=" * 90)
