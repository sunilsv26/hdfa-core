import asyncio
import aiohttp
from bs4 import BeautifulSoup

class HDFA_DocSpider:
    def __init__(self, urls):
        self.urls = urls
        self.harvested_pool = []

    async def fetch_and_clean(self, session, url):
        """Downloads a page and extracts pure code and text syntax context."""
        try:
            async with session.get(url, timeout=8) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 1. Harvest code fragments (syntax definitions)
                    code_snippets = [code.get_text().strip() for code in soup.find_all('code') if len(code.get_text().strip()) > 3]
                    
                    # 2. Harvest surrounding semantic documentation explanations
                    explanations = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 15]
                    
                    print(f"[SPIDER] Harvested data from: {url} | Found {len(code_snippets)} syntax blocks.")
                    return {"url": url, "snippets": code_snippets, "text": explanations}
        except Exception as e:
            print(f"[WARNING] Failed to stream {url}: {str(e)}")
            return None

    async def run(self):
        """Orchestrates concurrent async requests to stay lightweight on RAM."""
        connector = aiohttp.TCPConnector(limit_per_host=3)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.fetch_and_clean(session, url) for url in self.urls]
            results = await asyncio.gather(*tasks)
            self.harvested_pool = [r for r in results if r is not None]

# --- DAY 2 VALIDATION TEST ---
if __name__ == "__main__":
    print("Initializing Day 2: Async Documentation Spider...")
    
    # Target standard reference docs for HTML, CSS, JavaScript, and React
    sample_targets = [
        "https://react.dev",
        "https://react.dev",
        "https://mozilla.org",
        "https://mozilla.org"
    ]
    
    spider = HDFA_DocSpider(sample_targets)
    asyncio.run(spider.run())
    
    total_items = len(spider.harvested_pool)
    print(f"\n[SUCCESS] Day 2 complete. Streamed {total_items} complete doc matrices cleanly into memory.")
