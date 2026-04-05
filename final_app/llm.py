"""
Shared LLM instance used by all nodes.

Kept in one place so the model name can be changed without touching node files.
"""
from langchain_openai import ChatOpenAI

# gpt-4.1-mini balances quality and cost well for both planning and writing tasks
llm = ChatOpenAI(model="gpt-4.1-mini")
