import html
import re
import unicodedata
from datetime import datetime
from typing import List

import nltk as nltk
from scrapy.http import Response
from scrapy.spiders import CrawlSpider

from scrapers import items

nltk.download("punkt")


def clear_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\xad", "")
    text = text.replace("\u2009", "")
    text = re.sub("[\r\n]+", "\n", text)
    text = text.replace("\u00A0", "")
    text = text.replace("\u00a0", "")
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    return text.strip()


def fetch_post_texts(elements) -> List[str]:
    post_texts = []
    tags_to_separate = {"h2", "h3"}
    for e in elements:
        text = "".join(e.css("*::text").getall()).strip()
        if not text:
            continue
        if e.root.tag in tags_to_separate:
            text = "\n" + text
        post_texts.append(text)
    return post_texts


class TjournalSpider(CrawlSpider):
    name = "tjournal"
    allowed_domains = ["tjournal.ru"]
    start_urls = [f"https://tjournal.ru/post/{id}" for id in range(46592, 738368)]

    def parse_start_url(self, response: Response, **kwargs):
        basic_post_blocks = response.css(
            ".block-header, .block-text, .block-quote, .block-list, .raw-block"
        )
        post_elements = basic_post_blocks.css("h2, h3, p, ul li, ol li")
        post_texts = fetch_post_texts(post_elements)
        if not post_texts:
            return
        if post_texts[-1].startswith("#"):
            del post_texts[-1]
        document = []
        for paragraph in post_texts:
            labeled_sentences = [
                [clear_text(s), 0] for s in nltk.sent_tokenize(paragraph) if s.strip()
            ]
            labeled_sentences[0][1] = 1
            document.extend(labeled_sentences)

        item = items.PostItem()
        item["document"] = document
        yield document
