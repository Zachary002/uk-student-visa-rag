"""Turn raw HTML into clean, citation-ready Markdown.

Pipeline per page:
1. Parse HTML and locate the main content region (drop nav/header/footer/etc.).
2. Remove site-specific clutter (cookie banners, breadcrumbs, related-links,
   feedback widgets, the gov.uk "Contents" part-navigation).
3. Convert to Markdown (preserving headings, which help downstream chunking).
4. Trim everything before the page's own <h1> (pre-title banners/promos) and
   drop residual noise lines (share buttons, "print this page", etc.).
5. Prepend YAML front matter (title, source_url, retrieved_at) so the indexer
   can attach citation metadata to every chunk derived from this document.
"""
from __future__ import annotations

import re
from datetime import date
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify as to_md

# HTML tags that never hold primary content.
_STRIP_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "button", "svg", "iframe",
]

# Containers that hold the main article, tried in priority order.
_MAIN_SELECTORS = [
    "main", "#maincontent", "#main-content", "article", "[role=main]",
]

# Site-specific noise to delete from the main region (gov.uk / NHS / UKCISA).
_NOISE_SELECTORS = [
    ".gem-c-contents-list", ".gem-c-related-navigation", ".gem-c-contextual-sidebar",
    ".govuk-breadcrumbs", ".gem-c-feedback", "[data-module=feedback]",
    ".app-c-back-to-top", ".gem-c-phase-banner", ".gem-c-cookie-banner",
    ".nhsuk-breadcrumb", ".nhsuk-footer", "[data-uipath='page.lastreviewed']",
    ".breadcrumb", ".cookie-banner",
]

# Whole lines (ignoring leading #) that are boilerplate rather than content.
_NOISE_LINE_RE = re.compile(
    r"^\s*#{0,6}\s*("
    r"skip to (?:main )?content|skip contents|share|"
    r"view a printable version.*|get emails about this page|print this page|"
    r"is this page useful\??|related content|on this page|"
    r"open all|close all|read more|"           # accordion toggles / link CTAs
    r"sign up to our.*newsletter|stay in touch with ukcisa.*"  # newsletter CTA
    r")\s*:?\s*$",
    re.IGNORECASE,
)


def extract_title(soup: BeautifulSoup) -> str:
    """Best-effort page title: prefer <h1>, fall back to <title> (de-suffixed)."""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.string:
        return re.sub(r"\s*[-|–]\s*(GOV\.UK|NHS|UKCISA).*$", "",
                      soup.title.string).strip()
    return "Untitled"


def _select_main(soup: BeautifulSoup):
    for selector in _MAIN_SELECTORS:
        node = soup.select_one(selector)
        if node is not None:
            return node
    return soup.body or soup


def _strip_leading_chrome(markdown: str) -> str:
    """Drop pre-title chrome and the page's own leading H1 heading(s).

    Removes anything before the first H1 (breadcrumbs, promo banners) and then
    the leading H1(s) themselves — gov.uk guide parts carry two (the guide name
    and the part name). The canonical title is re-added from front matter, so
    keeping these would only duplicate it and create tiny heading-only chunks.
    If there is no H1, the text is returned unchanged to avoid eating content.
    """
    lines = markdown.split("\n")
    first_h1 = next(
        (i for i, line in enumerate(lines) if line.lstrip().startswith("# ")),
        None,
    )
    if first_h1 is None:
        return markdown.strip()

    i = first_h1
    while i < len(lines) and (
        lines[i].strip() == "" or lines[i].lstrip().startswith("# ")
    ):
        i += 1
    return "\n".join(lines[i:]).strip()


def _drop_noise_lines(markdown: str) -> str:
    return "\n".join(
        line for line in markdown.split("\n") if not _NOISE_LINE_RE.match(line)
    )


def _yaml_quote(value: str) -> str:
    """Quote a scalar so colons/quotes in titles don't break the YAML."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _friendly_title(source_url: str, title: str) -> str:
    """Make gov.uk guide parts distinguishable, e.g. 'Student visa: Money'.

    gov.uk guide parts all share the same <h1> (the guide name), so we append
    the URL's final path segment to keep citations precise. Other sites already
    have specific per-page <h1>s, so they are returned unchanged.
    """
    parts = urlparse(source_url)
    if parts.netloc.endswith("gov.uk"):
        segments = [s for s in parts.path.strip("/").split("/") if s]
        if len(segments) > 1:
            part = segments[-1].replace("-", " ").strip()
            part = part[:1].upper() + part[1:]
            return f"{title}: {part}"
    return title


def html_to_markdown(html: str) -> tuple[str, str]:
    """Return (extracted_title, clean_markdown_body) for a page's raw HTML."""
    soup = BeautifulSoup(html, "lxml")
    title = extract_title(soup)

    main = _select_main(soup)
    for tag in main.find_all(_STRIP_TAGS):
        tag.decompose()
    for selector in _NOISE_SELECTORS:
        for node in main.select(selector):
            node.decompose()

    # strip=["a", "img"] keeps link/image *text* but drops URL/asset clutter.
    markdown = to_md(str(main), heading_style="ATX", strip=["a", "img"])
    # Drop noise lines first: gov.uk wedges a "Skip contents" link between the
    # guide H1 and the part H1, which would otherwise stop _strip_leading_chrome
    # before it can remove the (redundant) second heading.
    markdown = _drop_noise_lines(markdown)
    markdown = _strip_leading_chrome(markdown)

    # Tidy whitespace: trailing spaces, then collapse 3+ blank lines.
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return title, markdown


def to_document(html: str, source_url: str) -> tuple[str, str]:
    """Produce (title, full_markdown_with_frontmatter) for one scraped page."""
    extracted_title, body = html_to_markdown(html)
    title = _friendly_title(source_url, extracted_title)
    front_matter = (
        "---\n"
        f"title: {_yaml_quote(title)}\n"
        f"source_url: {_yaml_quote(source_url)}\n"
        f"retrieved_at: {date.today().isoformat()}\n"
        "---\n\n"
    )
    return title, f"{front_matter}# {title}\n\n{body}\n"
