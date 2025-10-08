#!/usr/bin/env python3
"""
Google Gemini Client for Business Validation
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared/scripts'))

from base_llm_client import BaseLLMClient
import google.generativeai as genai

class GeminiClient(BaseLLMClient):
    """Google Gemini client for business validation"""
    
    def __init__(self):
        super().__init__("gemini")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def _get_api_key(self) -> str:
        """Get Google API key from environment"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return api_key
        
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to Google Gemini"""
        try:
            # Combine system and user prompts for Gemini
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            )
            
            response = self.model.generate_content(
                combined_prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")

if __name__ == "__main__":
    # Test the client
    client = GeminiClient()
    
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
