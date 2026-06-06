import requests
from bs4 import BeautifulSoup
import json
import time
import os

OUTPUT_FILE = "customer_reviews.json"
TOTAL_PAGES = 99
BASE_LIST_URL = "https://www.gsmarena.com/reviews.php3?iPage={}"
BASE_URL = "https://www.gsmarena.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def scrape_page(page_num):
    url = BASE_LIST_URL.format(page_num)
    print(f"Fetching page {page_num}/{TOTAL_PAGES}: {url}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")
    items = soup.select(".review-item.clearfix")
    page_data = []
    for item in items:
        title_tag = item.find("h3", class_="review-item-title")
        a_tag = item.find("a")
        if title_tag and a_tag:
            name = title_tag.get_text(strip=True)
            link = a_tag.get("href")
            if link and not link.startswith("http"):
                link = BASE_URL + link
            page_data.append({"name": name, "url": link})
    print(f"  Found {len(page_data)} reviews")
    return page_data


def scrape_review(url):
    print(f"  Fetching review: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        bodies = soup.find_all(class_="uopin")
        if bodies:
            texts = [body.get_text(separator="\n", strip=True) for body in bodies]
            return texts
        return None
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def main():
    # Load existing data if resuming
    all_reviews = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                all_reviews = json.load(f)
            print(f"Loaded {len(all_reviews)} existing reviews from {OUTPUT_FILE}")
        except Exception:
            all_reviews = {}

    # Collect all review links
    all_items = []
    for page in range(1, TOTAL_PAGES + 1):
        items = scrape_page(page)
        all_items.extend(items)
        time.sleep(1)

    print(f"\nTotal review items found: {len(all_items)}")

    # Scrape each review
    for i, item in enumerate(all_items):
        name = item["name"]
        url = item["url"]
        if name in all_reviews:
            continue
        print(f"[{i+1}/{len(all_items)}] {name}")
        review_text = scrape_review(url)
        if review_text:
            all_reviews[name] = review_text
        else:
            all_reviews[name] = []
            print(f"  No review body found for {name}")
        time.sleep(1)

        # Save periodically
        if (i + 1) % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_reviews, f, ensure_ascii=False, indent=2)
            print(f"  Saved checkpoint ({len(all_reviews)} reviews)")

    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_reviews, f, ensure_ascii=False, indent=2)
    print(f"\nDone! Saved {len(all_reviews)} reviews to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
