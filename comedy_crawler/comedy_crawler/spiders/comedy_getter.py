# -*- coding: utf-8 -*-
import scrapy
import re


class ComedyGetterSpider(scrapy.Spider):
    name = 'comedy-getter'
    allowed_domains = ['shakespeare.mit.edu']
    start_urls = ['http://shakespeare.mit.edu/']

    def parse(self, response):
        for comedy_link in response.xpath("//table[position()=2]/tr[position()=2]/td[position()=1]/a/@href"):
            yield response.follow(comedy_link.extract(), callback=self.resolve_index)

    def resolve_index(self, response):
        act_segments = response.xpath("//p")
        for act_segment in act_segments:
            for texts in act_segment.xpath("./text()").extract():
                if any([re.findall(r"Act \d+, Scene \d+:", text) is not None for text in texts]):
                    for scene_link in act_segment.xpath(".//a/@href"):
                        yield response.follow(scene_link.extract(), callback=self.extract_comedy)

    def extract_comedy(self, response):
        head = response.xpath("//h3/text()").extract_first()
        paragraphs = response.xpath("//h3/following-sibling::*")

        article = []
        for paragraph in paragraphs:
            tag_name = paragraph.xpath("name()").extract_first()
            text = " ".join(paragraph.xpath(".//text()").extract())
            text = re.sub(r" +", " ", text)
            if text.strip():
                article.append({tag_name: text})

        yield {
            "head": head,
            "article": article
        }
