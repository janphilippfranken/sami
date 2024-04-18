from typing import List, Dict
import os
import asyncio
from openai import AsyncAzureOpenAI 

class AsyncAzureChatLLM:
    """
    Wrapper for an (Async) Azure Chat Model.
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_version: str,
        retry_interval: int = 60,
        max_retries: int = 3,
    ):
        """
        Initializes AsyncAzureOpenAI client.
        """
        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint,
        )
        self.retry_interval = retry_interval
        self.max_retries = max_retries

    @property
    def llm_type(self):
        return "AsyncAzureOpenAI"

    async def __call__(self, messages: List[Dict[str, str]], **kwargs):
        """
        Make an async API call with retry functionality.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                response = await self.client.chat.completions.create(
                    messages=messages, **kwargs
                )
                return response
            except Exception as e:
                if retries < self.max_retries - 1:
                    print(f"Error occurred: {str(e)}. Retrying in {self.retry_interval} seconds...")
                    await asyncio.sleep(self.retry_interval)
                    retries += 1
                else:
                    return False