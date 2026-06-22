"""Unit tests for the HTML -> clean Markdown cleaner."""
from src.ingestion.cleaner import (
    _drop_noise_lines,
    html_to_markdown,
    to_document,
)


def test_html_to_markdown_strips_chrome_keeps_content() -> None:
    html = (
        "<html><body>"
        "<nav>menu links</nav>"
        "<main><h1>Student visa</h1><h1>Overview</h1>"
        "<p>You can work part time.</p>"
        "<script>track()</script></main>"
        "<footer>footer junk</footer>"
        "</body></html>"
    )
    title, markdown = html_to_markdown(html)

    assert title == "Student visa"
    assert "You can work part time." in markdown
    for noise in ("menu links", "footer junk", "track()"):
        assert noise not in markdown
    # Leading H1(s) are stripped (canonical title is re-added from front matter).
    assert not markdown.lstrip().startswith("#")


def test_drop_noise_lines_removes_boilerplate() -> None:
    md = "Real content\nSkip contents\nOpen all\nMore content\nRead more"
    out = _drop_noise_lines(md)
    assert "Real content" in out and "More content" in out
    for noise in ("Skip contents", "Open all", "Read more"):
        assert noise not in out


def test_to_document_adds_frontmatter_and_govuk_part_title() -> None:
    html = "<main><h1>Student visa</h1><p>Body about money.</p></main>"
    title, document = to_document(html, "https://www.gov.uk/student-visa/money")

    # gov.uk guide parts share one <h1>; we disambiguate with the URL part.
    assert title == "Student visa: Money"
    assert 'source_url: "https://www.gov.uk/student-visa/money"' in document
    # Exactly one H1 in the output (no duplicate heading).
    h1_lines = [line for line in document.splitlines() if line.startswith("# ")]
    assert h1_lines == ["# Student visa: Money"]


def test_to_document_keeps_specific_title_for_non_govuk() -> None:
    html = "<main><h1>Opening a bank account</h1><p>Choose a bank.</p></main>"
    title, _ = to_document(html, "https://www.ukcisa.org.uk/student-advice/finances/opening-a-bank-account/")
    assert title == "Opening a bank account"
