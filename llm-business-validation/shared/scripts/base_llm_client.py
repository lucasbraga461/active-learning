#!/usr/bin/env python3
"""
Base LLM Client for Business Validation
Provides a unified interface for different LLM providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BaseLLMClient(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.api_key = self._get_api_key()
        
    @abstractmethod
    def _get_api_key(self) -> str:
        """Get API key for this provider"""
        pass
        
    @abstractmethod
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to the LLM provider"""
        pass
        
    def validate_business(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a business record using the LLM
        
        Args:
            record: Business record with keys like 'business_name', 'address', 'city'
            
        Returns:
            Dictionary with validation results
        """
        try:
            system_prompt, user_prompt = self.build_validation_prompt(record)
            raw_response = self._call_api(system_prompt, user_prompt)
            parsed_result = self.parse_llm_response(raw_response)
            
            return {
                "input": record,
                "output": parsed_result,
                "raw_response": raw_response,
                "provider": self.provider_name,
                "attempts": 1
            }
            
        except Exception as e:
            return {
                "input": record,
                "output": {
                    "is_valid": False,
                    "confidence": 0.0,
                    "reasoning": f"Error during {self.provider_name} validation: {str(e)}"
                },
                "raw_response": "",
                "provider": self.provider_name,
                "attempts": 1,
                "error": str(e)
            }
    
    def build_validation_prompt(self, record: Dict[str, Any]) -> Tuple[str, str]:
        """Build system and user prompts for business validation."""
        system_prompt = (
            "You are a business validation expert. Your goal is to determine if a business is a VALID, "
            "OPERATING RESTAURANT that customers can currently visit and order food from. "
            "Use web search to gather information and make an informed decision based on what you find."
        )
        
        business_name = record.get('BUSINESS_NAME', '') or record.get('business_name', '')
        address = record.get('ADDRESS', '') or record.get('address', '')
        city = record.get('CITY', '') or record.get('city', '')
        
        search_query = f"{business_name} {address} {city}".strip()
        
        user_prompt = (
            f"SEARCH GOOGLE FOR: '{search_query}'\n\n"
            "Your goal: Determine if this is a VALID OPERATING RESTAURANT.\n\n"
            "IMPORTANT - MULTIPLE SEARCH STRATEGY:\n"
            "If your initial search doesn't return clear results, try these variations:\n"
            f"1. Just business name + city: '{business_name} {city}'\n"
            f"2. Address-focused search: '{address} {city} restaurant'\n"
            f"3. Partial name variations (remove special characters, try abbreviations)\n"
            f"4. Look for similar business names in the area\n\n"
            "A business is VALID if:\n"
            "✅ It's actually a restaurant, cafe, bakery, or food establishment (not a hotel, theater, office, etc.)\n"
            "✅ It's currently operating and serving customers (not permanently closed)\n"
            "✅ You can find evidence it exists and is legitimate (reviews, photos, business listings, etc.)\n\n"
            "Look for evidence such as:\n"
            "• Google Knowledge Panel with business info\n"
            "• Recent customer reviews\n"
            "• Food photos or menu images\n"
            "• Business hours and contact information\n"
            "• Social media presence\n"
            "• Any signs it's permanently closed\n\n"
            "CONFIDENCE GUIDELINES:\n"
            "• 90-100%: Strong evidence found (knowledge panel, many reviews, photos)\n"
            "• 70-89%: Good evidence but some uncertainty\n"
            "• 50-69%: Limited evidence or search limitations may apply\n"
            "• 30-49%: Weak evidence, likely invalid but not certain\n"
            "• 10-29%: Strong evidence it's invalid\n\n"
            "If NO results found, use 50-60% confidence and acknowledge search limitations.\n\n"
            "Output JSON only, exactly this schema:\n"
            "{\n"
            '  "is_valid": true | false,\n'
            '  "confidence": number,  // 0-100, how confident you are\n'
            '  "reasoning": "Detailed explanation of what you found and why you made this decision"\n'
            "}"
        )
        
        return system_prompt, user_prompt
    
    def parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response to extract validation results."""
        try:
            # Try to find JSON in the response
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_content[start_idx:end_idx]
            
            # Clean up common formatting issues
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            
            obj = json.loads(json_str)
            
            is_valid = bool(obj.get("is_valid", False))
            
            try:
                confidence = float(obj.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            
            confidence = max(0.0, min(100.0, confidence))
            reasoning = str(obj.get("reasoning", ""))[:1000]  # Limit reasoning length
            
            return {
                "is_valid": is_valid,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "reasoning": f"Failed to parse {self.provider_name} response: {str(e)}"
            }
