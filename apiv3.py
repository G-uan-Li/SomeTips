# -*- coding: utf-8 -*-
"""
@File    : apiv3.py
@Author  : qy
@Date    : 2025/11/17 17:16
"""
# -*- coding: utf-8 -*-
"""
@File    : api_async.py
@Author  : qy
@Date    : 2025/10/27 15:03
"""

import json
import httpx
import asyncio

API_URL = "https://mcc-pre.3xmt.com/gateway/ai-service/v1/chat/completions"
AUTH_TOKEN = "sk-bFcPcwS7J7oP6e8LGo"


async def call_local_model_async(prompt: str, stream: bool = False) -> str:
    """
    异步调用 deepseek-v3 模型
    """

    # 根据你的要求，payload 结构固定如下
    payload = {
        "stream": stream,
        "messages": [{"role": "user", "content": prompt}],
        "model": "deepseek-v3"
    }

    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json",
    }

    result_text = ""

    async with httpx.AsyncClient(timeout=180.0, verify=False) as client:
        if stream:
            async with client.stream("POST", API_URL, headers=headers, json=payload) as response:
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    line_data = line[6:]
                    if line_data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line_data)
                        delta = chunk["choices"][0].get("delta", {}).get("content", "")
                        result_text += delta
                    except Exception:
                        continue
        else:
            response = await client.post(API_URL, headers=headers, json=payload)
            result_json = response.json()
            result_text = result_json["choices"][0]["message"]["content"]

    return result_text.strip()


# 测试入口
# if __name__ == "__main__":
#     async def main():
#         text = await call_local_model_async("你好，介绍一下你自己", stream=False)
#         print(text)
#
#     asyncio.run(main())
