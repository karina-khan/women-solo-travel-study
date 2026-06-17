"""
r/femaletravels scraper — FULL VERSION
=======================================
Scrapes posts from January 2025 – May 2026.
Stops paginating when posts go older than start date.

Run scraper_test.py first to confirm the scraper is working.

Usage:
    python scripts/scraper_full.py

Output:
    data/raw/femaletravels_2025_2026.csv
"""

import time
import random
import os
import pandas as pd
from datetime import datetime, timezone
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# ── Config ─────────────────────────────────────────────────────────────────────

SUBREDDIT   = "femaletravels"
BASE_URL    = f"https://old.reddit.com/r/{SUBREDDIT}/"
OUTPUT_FILE = "data/raw/femaletravels_2025_2026.csv"

DATE_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
DATE_END   = datetime(2026, 5, 31, tzinfo=timezone.utc)

DELAY_MIN = 2.5
DELAY_MAX = 5.0

CHECKPOINT_EVERY = 100

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None


def parse_listing_page(html):
    soup = BeautifulSoup(html, "html.parser")
    posts = []

    for thing in soup.find_all("div", attrs={"data-type": "link"}):
        classes = thing.get("class", [])
        if "self" not in classes:
            continue
        if "stickied" in classes:
            continue

        post_id  = thing.get("data-fullname", "")
        title_el = thing.find("a", class_="title")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        url   = title_el.get("href", "")
        if url.startswith("/r/"):
            url = "https://old.reddit.com" + url

        time_el  = thing.find("time")
        date_str = time_el.get("datetime", "") if time_el else ""

        score_el = thing.find("div", class_="score")
        score    = score_el.get_text(strip=True) if score_el else ""

        posts.append({
            "post_id": post_id,
            "title":   title,
            "url":     url,
            "date":    date_str,
            "score":   score,
        })

    next_btn = soup.find("span", class_="next-button")
    next_url = None
    if next_btn:
        a = next_btn.find("a")
        if a:
            next_url = a.get("href")

    return posts, next_url


def scrape_post_page(page, url):
    try:
        page.goto(url, timeout=20000)
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        body_text = ""
        usertext  = soup.find("div", class_="usertext-body")
        if usertext:
            for tag in usertext.find_all(["a", "script"]):
                tag.decompose()
            body_text = usertext.get_text(separator=" ", strip=True)
        if body_text.strip().lower() in ["[removed]", "[deleted]", ""]:
            body_text = ""

        comments = []
        for c in soup.find_all("div", class_="comment"):
            if c.get("data-depth", "1") != "0":
                continue
            c_body = c.find("div", class_="usertext-body")
            if not c_body:
                continue
            c_text = c_body.get_text(separator=" ", strip=True)
            if c_text.lower() not in ["[removed]", "[deleted]", ""]:
                comments.append(c_text)
            if len(comments) >= 10:
                break

        num_comments = ""
        cl = soup.find("a", class_="comments")
        if cl:
            num_comments = cl.get_text(strip=True)

        return {
            "body":         body_text,
            "comments":     " ||| ".join(comments),
            "num_comments": num_comments,
        }

    except Exception as e:
        print(f"    ⚠ Error: {e}")
        return {"body": "", "comments": "", "num_comments": ""}


def save_checkpoint(posts, output_file):
    df = pd.DataFrame(posts)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"  💾 Checkpoint saved ({len(df)} posts → {output_file})")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_full_scrape():
    os.makedirs("data/raw", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"r/{SUBREDDIT} — FULL SCRAPE")
    print(f"Date range: {DATE_START.date()} → {DATE_END.date()}")
    print(f"{'='*60}\n")

    all_posts   = []
    current_url = BASE_URL
    page_num    = 1
    stop        = False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        context.add_cookies([{
            'name':   'redesign_optout',
            'value':  'true',
            'domain': '.reddit.com',
            'path':   '/'
        }])
        page = context.new_page()

        # Phase 1: collect post stubs within date range
        print("Phase 1 — Collecting post URLs within date range")
        print("-" * 40)

        post_stubs = []

        while current_url and not stop:
            print(f"  Listing page {page_num} ({len(post_stubs)} posts so far)...")
            page.goto(current_url, timeout=20000)
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

            html = page.content()
            listing_posts, next_url = parse_listing_page(html)

            if not listing_posts:
                print("  No posts found on this page — stopping")
                break

            for stub in listing_posts:
                post_date = parse_date(stub["date"])

                if post_date and post_date > DATE_END:
                    continue

                if post_date and post_date < DATE_START:
                    print(f"  Reached posts older than {DATE_START.date()} — stopping")
                    stop = True
                    break

                post_stubs.append(stub)

            print(f"  → {len(listing_posts)} on page, {len(post_stubs)} in range so far")
            current_url = next_url
            page_num   += 1

        print(f"\n✓ {len(post_stubs)} posts in date range\n")

        # Phase 2: scrape each post
        print("Phase 2 — Scraping post bodies and comments")
        print("-" * 40)

        for i, stub in enumerate(post_stubs):
            title_preview = stub["title"][:55] + "..." if len(stub["title"]) > 55 else stub["title"]
            print(f"  [{i+1:04d}/{len(post_stubs)}] {title_preview}")

            post_data = scrape_post_page(page, stub["url"])
            all_posts.append({**stub, **post_data})

            if (i + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(all_posts, OUTPUT_FILE)

        browser.close()

    df = pd.DataFrame(all_posts)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n{'='*60}")
    print("SCRAPE COMPLETE")
    print(f"{'='*60}")
    print(f"Total posts:               {len(df)}")
    if len(df) > 0:
        has_body     = (df["body"].str.strip().str.len() > 0).sum()
        has_comments = (df["comments"].str.strip().str.len() > 0).sum()
        print(f"Posts with body text:      {has_body} ({has_body/len(df)*100:.0f}%)")
        print(f"Posts with comments:       {has_comments} ({has_comments/len(df)*100:.0f}%)")
        print(f"Date range in data:        {df['date'].min()} → {df['date'].max()}")
    print(f"Saved to:                  {OUTPUT_FILE}")


if __name__ == "__main__":
    run_full_scrape()