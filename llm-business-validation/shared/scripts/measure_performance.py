#!/usr/bin/env python3
"""
Measure API Speed and Cost for LLM Business Validation
"""

import sys
import os
import time
import pandas as pd
from typing import Dict, Any

# Add experiment paths
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '../../experiments/openai'))
sys.path.append(os.path.join(current_dir, '../../experiments/perplexity'))
sys.path.append(os.path.join(current_dir, '../../experiments/gemini'))

from openai_client import OpenAIClient
from perplexity_client import PerplexityClient
from gemini_client import GeminiClient

# API Pricing (as of September 2024)
PRICING = {
    'openai': {
        'model': 'gpt-4o',
        'input_cost_per_1k': 0.0025,   # $2.50 per 1M tokens
        'output_cost_per_1k': 0.01     # $10.00 per 1M tokens
    },
    'perplexity': {
        'model': 'sonar',
        'cost_per_request': 0.005       # $5.00 per 1K requests (includes web search)
    },
    'gemini': {
        'model': 'gemini-1.5-pro',
        'input_cost_per_1k': 0.00125,  # $1.25 per 1M tokens
        'output_cost_per_1k': 0.005    # $5.00 per 1M tokens
    }
}

def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token"""
    return len(text) // 4

def measure_single_request(client, record: Dict[str, Any], provider: str) -> Dict[str, Any]:
    """Measure speed and estimate cost for a single request"""
    start_time = time.time()
    
    try:
        result = client.validate_business(record)
        end_time = time.time()
        
        # Calculate response time
        response_time = end_time - start_time
        
        # Estimate cost
        if provider == 'perplexity':
            # Perplexity charges per request
            estimated_cost = PRICING[provider]['cost_per_request']
        else:
            # OpenAI and Gemini charge per token
            # Estimate input tokens (business name + address + city + prompt)
            input_text = f"{record.get('business_name', '')} {record.get('address', '')} {record.get('city', '')}"
            input_tokens = estimate_tokens(input_text) + 500  # Add ~500 for system prompt
            
            # Estimate output tokens from reasoning length
            output_tokens = estimate_tokens(result['output'].get('reasoning', '')) + 50  # Add buffer
            
            input_cost = (input_tokens / 1000) * PRICING[provider]['input_cost_per_1k']
            output_cost = (output_tokens / 1000) * PRICING[provider]['output_cost_per_1k']
            estimated_cost = input_cost + output_cost
        
        return {
            'success': True,
            'response_time': response_time,
            'estimated_cost': estimated_cost,
            'result': result
        }
        
    except Exception as e:
        end_time = time.time()
        return {
            'success': False,
            'response_time': end_time - start_time,
            'estimated_cost': 0.0,
            'error': str(e)
        }

def main():
    # Test records
    test_records = [
        {"business_name": "STARBUCKS", "address": "157 LAFAYETTE STREET", "city": "New York"},
        {"business_name": "MCDONALD'S", "address": "151 WEST 34 STREET", "city": "New York"},
        {"business_name": "SHAKE SHACK", "address": "820 WASHINGTON STREET", "city": "New York"}
    ]
    
    clients = {
        'openai': OpenAIClient(),
        'perplexity': PerplexityClient(),
        'gemini': GeminiClient()
    }
    
    results = {}
    
    print("🚀 Measuring API Performance and Cost...")
    print("=" * 60)
    
    for provider, client in clients.items():
        print(f"\n📊 Testing {provider.upper()}...")
        
        provider_results = []
        total_time = 0
        total_cost = 0
        successful_requests = 0
        
        for i, record in enumerate(test_records):
            print(f"   Request {i+1}/3: {record['business_name']}")
            
            measurement = measure_single_request(client, record, provider)
            provider_results.append(measurement)
            
            if measurement['success']:
                total_time += measurement['response_time']
                total_cost += measurement['estimated_cost']
                successful_requests += 1
                print(f"      ✅ {measurement['response_time']:.2f}s, ${measurement['estimated_cost']:.4f}")
            else:
                print(f"      ❌ Failed: {measurement['error']}")
            
            # Add delay between requests to be respectful
            if i < len(test_records) - 1:
                time.sleep(1)
        
        # Calculate averages
        avg_time = total_time / max(successful_requests, 1)
        avg_cost = total_cost / max(successful_requests, 1)
        
        results[provider] = {
            'avg_response_time': avg_time,
            'avg_cost_per_request': avg_cost,
            'successful_requests': successful_requests,
            'total_requests': len(test_records),
            'success_rate': successful_requests / len(test_records) * 100
        }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE & COST COMPARISON")
    print("=" * 60)
    print(f"{'Provider':<12} {'Avg Time':<10} {'Avg Cost':<12} {'Success Rate':<12}")
    print("-" * 60)
    
    for provider, data in results.items():
        print(f"{provider.capitalize():<12} {data['avg_response_time']:.2f}s{'':<4} "
              f"${data['avg_cost_per_request']:.4f}{'':<4} {data['success_rate']:.1f}%")
    
    # Extrapolate to 115 samples
    print(f"\n💰 ESTIMATED COST FOR 115 SAMPLES:")
    print("-" * 40)
    for provider, data in results.items():
        total_cost_115 = data['avg_cost_per_request'] * 115
        total_time_115 = (data['avg_response_time'] * 115) / 60  # Convert to minutes
        print(f"{provider.capitalize():<12} ${total_cost_115:.2f} ({total_time_115:.1f} minutes)")
    
    return results

if __name__ == "__main__":
    main()
