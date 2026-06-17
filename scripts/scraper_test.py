"""
r/femaletravels scraper — TEST VERSION
=======================================
Scrapes 50 posts from old.reddit.com/r/femaletravels
Extracts: title, body text, date, top comments, score

Run this first to confirm data quality before the full scrape.

Usage:
    python scripts/scraper_test.py

Output:
    data/raw/test_scrape.csv
"""

import time
import random
import os
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# ── Config ─────────────────────────────────────────────────────────────────────

SUBREDDIT   = "femaletravels"
BASE_URL    = f"https://old.reddit.com/r/{SUBREDDIT}/"
TEST_LIMIT  = 50
OUTPUT_FILE = "data/raw/test_scrape.csv"

DELAY_MIN = 2.0
DELAY_MAX = 4.0

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ── Listing parser ─────────────────────────────────────────────────────────────

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


# ── Post scraper ───────────────────────────────────────────────────────────────

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
        print(f"    ⚠ Error on {url}: {e}")
        return {"body": "", "comments": "", "num_comments": ""}


# ── Main ───────────────────────────────────────────────────────────────────────

def run_test():
    os.makedirs("data/raw", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"r/{SUBREDDIT} — TEST SCRAPE ({TEST_LIMIT} posts)")
    print(f"{'='*60}\n")

    all_posts   = []
    current_url = BASE_URL

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

        # Phase 1: collect post URLs
        print("Phase 1 — Collecting post URLs from listing pages")
        print("-" * 40)

        post_stubs = []
        page_num   = 1

        while len(post_stubs) < TEST_LIMIT and current_url:
            print(f"  Listing page {page_num}: {current_url[:80]}")
            page.goto(current_url, timeout=20000)
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

            html = page.content()
            posts, next_url = parse_listing_page(html)

            post_stubs.extend(posts)
            print(f"  → {len(posts)} posts found (running total: {len(post_stubs)})")

            current_url = next_url
            page_num   += 1

            if len(post_stubs) >= TEST_LIMIT:
                break

        post_stubs = post_stubs[:TEST_LIMIT]
        print(f"\n✓ Collected {len(post_stubs)} post URLs\n")

        # Phase 2: scrape each post
        print("Phase 2 — Scraping post bodies and comments")
        print("-" * 40)

        for i, stub in enumerate(post_stubs):
            title_preview = stub["title"][:55] + "..." if len(stub["title"]) > 55 else stub["title"]
            print(f"  [{i+1:02d}/{len(post_stubs)}] {title_preview}")
            post_data = scrape_post_page(page, stub["url"])
            all_posts.append({**stub, **post_data})

        browser.close()

    df = pd.DataFrame(all_posts)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total posts scraped:       {len(df)}")

    if len(df) > 0:
        has_body     = (df["body"].str.strip().str.len() > 0).sum()
        has_comments = (df["comments"].str.strip().str.len() > 0).sum()
        has_neither  = ((df["body"].str.len() == 0) & (df["comments"].str.len() == 0)).sum()

        print(f"Posts with body text:      {has_body} ({has_body/len(df)*100:.0f}%)")
        print(f"Posts with comments:       {has_comments} ({has_comments/len(df)*100:.0f}%)")
        print(f"Posts with neither:        {has_neither} ({has_neither/len(df)*100:.0f}%)")
        print(f"Saved to:                  {OUTPUT_FILE}")

        print(f"\n── Sample titles ──")
        for t in df["title"].head(5):
            print(f"  • {t}")

        print(f"\n── First post with body text ──")
        sample_body = df[df["body"].str.len() > 0]["body"]
        if not sample_body.empty:
            print(f"  {sample_body.iloc[0][:400]}")
        else:
            print("  ⚠ No body text found")

        print(f"\n── First post with comments ──")
        sample_comments = df[df["comments"].str.len() > 0]["comments"]
        if not sample_comments.empty:
            print(f"  {sample_comments.iloc[0].split(' ||| ')[0][:400]}")
        else:
            print("  ⚠ No comments found")


if __name__ == "__main__":
    run_test()