from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()

tool = TavilySearch(max_results=2)
results = tool.invoke({"query": 'ChatGPT version releases and updates from 2022 to 2026'})
print(results)