# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


class ComedyCrawlerPipeline(object):
    def process_item(self, item, spider):
        content = ""

        head = item["head"]
        content += f"<head> {head} </head>\n"

        articles = item["article"]
        for article in articles:
            for k, v in article.items():
                if k != "table":
                    content += f"<{k}> {v.strip()} </{k}>\n"

        with open("../data/full-text.txt", "a") as f:
            f.write(content)
