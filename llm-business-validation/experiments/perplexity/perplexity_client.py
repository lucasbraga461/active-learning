#!/usr/bin/env python3
"""
Perplexity Client for Business Validation
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared/scripts'))

from base_llm_client import BaseLLMClient
from openai import OpenAI

class PerplexityClient(BaseLLMClient):
    """Perplexity AI client for business validation"""
    
    def __init__(self):
        super().__init__("perplexity")
        # Perplexity uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai"
        )
        
    def _get_api_key(self) -> str:
        """Get Perplexity API key from environment"""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment variables")
        return api_key
        
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to Perplexity"""
        try:
            response = self.client.chat.completions.create(
                model="sonar",  # Perplexity's web-search enabled model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Perplexity API call failed: {str(e)}")

if __name__ == "__main__":
    # Test the client
    client = PerplexityClient()
    
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
