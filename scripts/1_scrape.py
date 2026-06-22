"""P1 entrypoint — build the knowledge base from official sources.

Scrapes a curated, verified list of gov.uk / NHS / UKCISA pages (robots.txt
checked, rate-limited), cleans each to Markdown, and writes them to
``data/processed/``.

Usage:
    python scripts/1_scrape.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

# Make ``src`` importable when run as a script (python scripts/1_scrape.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings                                  # noqa: E402
from src.ingestion.cleaner import to_document                    # noqa: E402
from src.ingestion.scraper import RobotsCache, fetch, make_session  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("scrape")

# Curated corpus: 18 official pages covering visas, work rights, finances, NHS.
# Each URL was verified (HTTP 200) and checked against robots.txt during P1.
SOURCES: list[str] = [
    # --- gov.uk: Student visa ---
    "https://www.gov.uk/student-visa",                  # overview: work hours, stay, do's/don'ts
    "https://www.gov.uk/student-visa/money",            # financial requirement
    "https://www.gov.uk/student-visa/knowledge-of-english",
    "https://www.gov.uk/student-visa/extend-your-visa",  # renewing / extending
    "https://www.gov.uk/student-visa/switch-to-this-visa",
    "https://www.gov.uk/student-visa/family-members",   # dependants
    "https://www.gov.uk/student-visa/course",
    # --- gov.uk: Graduate visa ---
    "https://www.gov.uk/graduate-visa",                 # overview: work rights, duration
    "https://www.gov.uk/graduate-visa/how-much-it-costs",
    "https://www.gov.uk/graduate-visa/course-you-studied",  # eligibility
    # --- gov.uk: Immigration Health Surcharge (links to NHS) ---
    "https://www.gov.uk/healthcare-immigration-application",
    # --- NHS ---
    "https://www.nhs.uk/nhs-services/gps/how-to-register-with-a-gp-surgery/",
    # Non-EEA migrants who paid the IHS (most international students) use the NHS
    # like residents — this sub-page spells that out (the parent was just a hub).
    "https://www.nhs.uk/nhs-services/visiting-or-moving-to-england/moving-to-england-from-outside-the-european-economic-area-eea/",
    # --- UKCISA (substantive sub-pages; the section landing pages are just hubs) ---
    "https://www.ukcisa.org.uk/student-advice/working/student-work/",       # term-time work-hour limits (20/10h)
    "https://www.ukcisa.org.uk/student-advice/working/working-in-the-uk/",  # right to work, tax, finding a job
    "https://www.ukcisa.org.uk/student-advice/finances/opening-a-bank-account/",  # banking
    "https://www.ukcisa.org.uk/student-advice/visas-and-immigration/student-immigration-the-basics/",
    "https://www.ukcisa.org.uk/student-advice/visas-and-immigration/student-route-applying-in-the-uk/",
    "https://www.ukcisa.org.uk/student-advice/life-in-the-uk/healthcare/",
]


def url_to_filename(url: str) -> str:
    """Stable, readable filename, e.g. gov-uk__student-visa__money.md"""
    parts = urlparse(url)
    host = parts.netloc.replace("www.", "").replace(".", "-")
    path = parts.path.strip("/").replace("/", "__") or "index"
    return f"{host}__{path}.md"


def main() -> int:
    settings.ensure_dirs()
    robots = RobotsCache()
    session = make_session()

    saved, skipped = 0, 0
    for url in SOURCES:
        result = fetch(url, robots, session)
        if not result.ok or result.html is None:
            logger.warning("SKIP  %s  (%s)", url, result.reason)
            skipped += 1
            continue

        title, document = to_document(result.html, url)
        out_path = settings.paths.data_processed / url_to_filename(url)
        out_path.write_text(document, encoding="utf-8")
        logger.info("SAVED %-45s <- %s", out_path.name, title)
        saved += 1

    logger.info("Done. %d saved, %d skipped (of %d).", saved, skipped, len(SOURCES))
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
