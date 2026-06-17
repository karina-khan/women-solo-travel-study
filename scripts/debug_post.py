"""
Debug script — visits one post URL and saves the raw HTML so we can
inspect what Reddit is actually serving to the scraper.
"""
import time
from playwright.sync_api import sync_playwright

URL = "https://old.reddit.com/r/femaletravels/comments/1tdxlp4/solo_luxuryresort_travel/"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"
    )
    page = context.new_page()
    page.goto(URL, timeout=20000)
    time.sleep(3)
    html = page.content()
    browser.close()

with open("data/raw/debug_post.html", "w") as f:
    f.write(html)

print(f"Saved {len(html)} chars to data/raw/debug_post.html")
print("\nFirst 500 chars:")
print(html[:500])