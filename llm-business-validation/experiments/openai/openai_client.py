#!/usr/bin/env python3
"""
OpenAI Client for Business Validation
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared/scripts'))

from base_llm_client import BaseLLMClient
from openai import OpenAI

class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client for business validation"""
    
    def __init__(self):
        super().__init__("openai")
        self.client = OpenAI(api_key=self.api_key)
        
    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return api_key
        
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # or gpt-4o-mini for faster/cheaper
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")

if __name__ == "__main__":
    # Test the client
    client = OpenAIClient()
    
    test_record = {
        "business_name": "STARBUCKS",
        "address": "157 LAFAYETTE STREET", 
        "city": "New York"
    }
    
    result = client.validate_business(test_record)
    print(f"Provider: {result['provider']}")
    print(f"Valid: {result['output']['is_valid']}")
    print(f"Confidence: {result['output']['confidence']}")
    print(f"Reasoning: {result['output']['reasoning'][:100]}...")
