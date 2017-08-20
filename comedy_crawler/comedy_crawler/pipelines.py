# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


class ComedyCrawlerPipeline(object):
    def process_item(self, item, spider):
        content = "\n".join(item["article"])
        with open("../data/full-text.txt", "a") as f:
            f.write(content)
