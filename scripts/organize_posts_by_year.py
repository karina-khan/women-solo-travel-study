"""
Organize Reddit posts CSV by year into separate files.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Read the posts CSV
posts_file = Path(__file__).parent.parent / "arctic-shift/data/r_solofemaletravellers_2018-01-01_to_2026-06-30_posts.csv"
output_dir = Path(__file__).parent.parent / "data/processed"

output_dir.mkdir(parents=True, exist_ok=True)

# Dictionary to store posts by year
posts_by_year = defaultdict(list)

# Read the CSV and organize by year
with open(posts_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        # created_utc is a Unix timestamp
        created_utc = int(row.get('created_utc', 0))
        dt = datetime.fromtimestamp(created_utc)
        year = dt.year
        posts_by_year[year].append(row)

print(f"Total posts: {sum(len(posts) for posts in posts_by_year.values())}")
print(f"Years with data: {sorted(posts_by_year.keys())}")

# Write each year to a separate CSV
for year in sorted(posts_by_year.keys()):
    posts = posts_by_year[year]
    output_file = output_dir / f"solofemaletravellers_{year}_posts.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(posts)

    print(f"  {year}: {len(posts)} posts → {output_file}")

# Also create a summary CSV
summary_file = output_dir / "solofemaletravellers_summary.csv"
with open(summary_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Year', 'Post Count'])
    for year in sorted(posts_by_year.keys()):
        writer.writerow([year, len(posts_by_year[year])])

print(f"\nSummary saved → {summary_file}")
