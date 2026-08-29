import requests
from bs4 import BeautifulSoup
import json, time, os

OUTPUT_FILE = "customer_reviews.json"
TOTAL_PAGES = 99
BASE_LIST_URL = "https://www.gsmarena.com/reviews.php3?iPage={}"
BASE_URL = "https://www.gsmarena.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def scrape_page(page_num):
    r = requests.get(BASE_LIST_URL.format(page_num), headers=HEADERS, timeout=30)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")
    items = soup.select(".review-item.clearfix")
    results = []
    for item in items:
        title_tag = item.find("h3", class_="review-item-title")
        a_tag = item.find("a")
        if title_tag and a_tag:
            link = a_tag.get("href", "")
            if link and not link.startswith("http"):
                link = BASE_URL + link
            results.append({"name": title_tag.get_text(strip=True), "url": link})
    print(f"Page {page_num}: {len(results)} reviews")
    return results


def scrape_review(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.encoding = "utf-8"
        bodies = BeautifulSoup(r.text, "html.parser").find_all(class_="uopin")
        return [b.get_text(separator="\n", strip=True) for b in bodies] if bodies else None
    except Exception as e:
        print(f"  Error: {e}")
        return None


# Load existing data if resuming
all_reviews = {}
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            all_reviews = json.load(f)
        print(f"Loaded {len(all_reviews)} existing reviews")
    except Exception:
        pass

# Collect all review links
all_items = []
for page in range(1, TOTAL_PAGES + 1):
    all_items.extend(scrape_page(page))
    time.sleep(1)
print(f"\nTotal review items: {len(all_items)}")

# Scrape each review
for i, item in enumerate(all_items):
    name, url = item["name"], item["url"]
    if name in all_reviews:
        continue
    print(f"[{i+1}/{len(all_items)}] {name}")
    all_reviews[name] = scrape_review(url) or []
    time.sleep(1)
    if (i + 1) % 10 == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_reviews, f, ensure_ascii=False, indent=2)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_reviews, f, ensure_ascii=False, indent=2)
print(f"\nDone! Saved {len(all_reviews)} reviews to {OUTPUT_FILE}")
