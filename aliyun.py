import re
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

# 配置星火的接口信息
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
SPARKAI_APP_ID = '702e7404'  # 替换为你的 APP ID
SPARKAI_API_SECRET = 'ZmY1MWY5OGJmYThjYWEzNDNjNjY3YWNl'  # 替换为你的 API Secret
SPARKAI_API_KEY = '937d0855d56fd12639748138d6c15d33'  # 替换为你的 API Key
SPARKAI_DOMAIN = 'generalv3.5'

# 初始化星火模型
spark = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN,
    streaming=False,
)

# 构建提问内容
def get_paragraph_prompt(paragraph):
    paragraph_prompt = f"""
    你好，你是一个专门从输入的段落中抽取实体和关系的专家，请从下面的段落中提取所有可能的三元组。
    输入：{paragraph}
    输出：
    """
    return paragraph_prompt

# 请求星火模型
def ask_spark(paragraph_prompt):
    messages = [ChatMessage(role="user", content=paragraph_prompt)]
    handler = ChunkPrintHandler()
    result = spark.generate([messages], callbacks=[handler])
    return result

def one_turn(paragraph):
    paragraph_prompt = get_paragraph_prompt(paragraph)
    spark_response = ask_spark(paragraph_prompt)
    return spark_response

paragraph = "李华在2022年成为了华为的首席技术官。此前，他在微软担任过高级工程师。李华毕业于清华大学，并且拥有计算机科学博士学位。"
response = one_turn(paragraph)
print(response)
