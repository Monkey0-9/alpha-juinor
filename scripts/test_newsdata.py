import os
import sys
from dotenv import load_dotenv

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from data.providers.news_provider import NewsDataProvider

def test_news_provider():
    load_dotenv()
    provider = NewsDataProvider()
    if getattr(provider, 'disabled', False):
        print("FAIL: NewsDataProvider is disabled. Check API key.")
        return

    print("Fetching news for AAPL...")
    news = provider.get_news_data("AAPL", start_date="2026-01-01")
    if news:
        print(f"SUCCESS: Fetched {len(news)} articles.")
        for i, item in enumerate(news[:3]):
            print(f"{i+1}. {item['title']} (Source: {item['source']})")
    else:
        print("FAIL: No news items fetched.")

if __name__ == "__main__":
    test_news_provider()
