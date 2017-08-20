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
        act_segments = response.xpath("//body/p[position()>2]/a")
        for act_segment in act_segments:
            scene_link = act_segment.xpath("./@href")
            yield response.follow(scene_link.extract_first(), callback=self.extract_comedy)

    def extract_comedy(self, response):
        paragraphs = response.xpath("//blockquote/a")

        article = []
        for paragraph in paragraphs:
            text = " ".join(paragraph.xpath(".//text()").extract())
            text = re.sub(r" +", " ", text)
            if text.strip():
                article.append(text)

        yield {
            "article": article
        }
