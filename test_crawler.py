# test_crawler.py
from rag_service import crawl

def test_crawl_example():
    # Quick sanity: crawl 1 page from example.com
    res = crawl("https://example.com", max_pages=1, max_depth=1, crawl_delay_ms=0)
    assert isinstance(res, dict)
    assert "page_count" in res
    assert res["page_count"] >= 1
