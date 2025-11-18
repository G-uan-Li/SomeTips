# -*- coding: utf-8 -*-
"""
@File    : max-token.py
@Author  : qy
@Date    : 2025/11/7 18:51
"""
import json

# 假设你的 JSON 文件名为 data.json
with open("C:\\Users\\qy\\Desktop\\2025work\\微调数据\\zhengqi_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

max_len = 0
max_item = None

for item in data:
    combined = item.get("instruction", "") + item.get("input", "") + item.get("output", "")
    length = len(combined)
    if length > max_len:
        max_len = length
        max_item = item

print("最大长度：", max_len)
# 如果想看是哪一条字典
# print(max_item)
