#!/usr/bin/env python3
"""
Multi-LLM Business Validation Runner
Runs the same validation across multiple LLM providers for comparison
"""

import sys
import os
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import argparse
from tqdm import tqdm

# Add experiment paths
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '../../experiments/openai'))
sys.path.append(os.path.join(current_dir, '../../experiments/perplexity'))
sys.path.append(os.path.join(current_dir, '../../experiments/gemini'))

from openai_client import OpenAIClient
from perplexity_client import PerplexityClient
from gemini_client import GeminiClient

class MultiLLMValidator:
    """Run business validation across multiple LLM providers"""
    
    def __init__(self, providers: List[str] = None):
        """
        Initialize with specified providers
        
        Args:
            providers: List of provider names ('openai', 'perplexity', 'gemini')
        """
        if providers is None:
            providers = ['openai', 'perplexity', 'gemini']
            
        self.clients = {}
        
        # Initialize available clients
        for provider in providers:
            try:
                if provider == 'openai':
                    self.clients['openai'] = OpenAIClient()
                elif provider == 'perplexity':
                    self.clients['perplexity'] = PerplexityClient()
                elif provider == 'gemini':
                    self.clients['gemini'] = GeminiClient()
                print(f"✅ {provider.title()} client initialized")
            except Exception as e:
                print(f"❌ Failed to initialize {provider}: {e}")
                
        if not self.clients:
            raise Exception("No LLM clients could be initialized. Check your API keys.")
    
    def validate_single_record(self, record: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """Validate a single business record with specified provider"""
        try:
            client = self.clients[provider]
            result = client.validate_business(record)
            return result
        except Exception as e:
            return {
                "input": record,
                "output": {
                    "is_valid": False,
                    "confidence": 0.0,
                    "reasoning": f"Error with {provider}: {str(e)}"
                },
                "provider": provider,
                "error": str(e)
            }
    
    def run_comparison(self, input_csv: str, max_workers: int = 3, limit: int = None) -> Dict[str, List[Dict]]:
        """
        Run validation comparison across all providers
        
        Args:
            input_csv: Path to input CSV file
            max_workers: Number of concurrent workers per provider
            limit: Limit number of records to process (for testing)
            
        Returns:
            Dictionary with results for each provider
        """
        # Load data
        df = pd.read_csv(input_csv)
        if limit:
            df = df.head(limit)
            
        records = df.to_dict('records')
        
        print(f"🚀 Running validation on {len(records)} records across {len(self.clients)} providers...")
        
        all_results = {}
        
        # Run each provider
        for provider_name, client in self.clients.items():
            print(f"\n📊 Running {provider_name.title()} validation...")
            
            results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_record = {
                    executor.submit(self.validate_single_record, record, provider_name): record 
                    for record in records
                }
                
                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_record), 
                                 total=len(records), 
                                 desc=f"{provider_name.title()}"):
                    result = future.result()
                    results.append(result)
            
            all_results[provider_name] = results
            print(f"✅ {provider_name.title()} completed: {len(results)} records")
        
        return all_results
    
    def save_results(self, results: Dict[str, List[Dict]], base_output_dir: str = "../../experiments"):
        """Save results for each provider in their respective folders"""
        
        for provider, provider_results in results.items():
            # Create provider-specific results directory
            provider_results_dir = os.path.join(base_output_dir, provider, "results")
            os.makedirs(provider_results_dir, exist_ok=True)
            
            # Determine file naming based on number of records
            num_records = len(provider_results)
            
            # Save JSONL
            jsonl_path = os.path.join(provider_results_dir, f"{provider}_{num_records}_results.jsonl")
            with open(jsonl_path, 'w') as f:
                for result in provider_results:
                    f.write(json.dumps(result) + '\n')
            
            # Save CSV summary
            csv_data = []
            for result in provider_results:
                input_data = result['input']
                output_data = result['output']
                
                csv_row = {
                    'business_name': input_data.get('business_name', ''),
                    'address': input_data.get('address', ''),
                    'city': input_data.get('city', ''),
                    'is_valid': output_data.get('is_valid', False),
                    'confidence': output_data.get('confidence', 0.0),
                    'reasoning': output_data.get('reasoning', ''),
                    'provider': provider
                }
                csv_data.append(csv_row)
            
            csv_path = os.path.join(provider_results_dir, f"{provider}_{num_records}_results.csv")
            pd.DataFrame(csv_data).to_csv(csv_path, index=False)
            
            print(f"💾 {provider.title()} results saved:")
            print(f"   📄 {jsonl_path}")
            print(f"   📊 {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-LLM Business Validation Comparison")
    parser.add_argument("--input-csv", required=True, help="Input CSV file with business data")
    parser.add_argument("--providers", nargs='+', default=['openai', 'perplexity', 'gemini'],
                       choices=['openai', 'perplexity', 'gemini'], 
                       help="LLM providers to use")
    parser.add_argument("--max-workers", type=int, default=3, help="Max concurrent workers per provider")
    parser.add_argument("--limit", type=int, help="Limit number of records (for testing)")
    parser.add_argument("--output-dir", default="../../experiments", help="Base output directory for experiments")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = MultiLLMValidator(providers=args.providers)
    
    # Run comparison
    results = validator.run_comparison(
        input_csv=args.input_csv,
        max_workers=args.max_workers,
        limit=args.limit
    )
    
    # Save results
    validator.save_results(results, output_dir=args.output_dir)
    
    print(f"\n🎉 Multi-LLM comparison completed!")
    print(f"📁 Results saved in: {args.output_dir}/")

if __name__ == "__main__":
    main()
