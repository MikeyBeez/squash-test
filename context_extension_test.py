#!/usr/bin/env python3
"""
Context Window Extension Experiment

Tests whether pretrained language models can handle context windows
larger than their training size without catastrophic failure.

This experiment is critical for understanding if semantic compression
can work by simply extending context rather than requiring full retraining.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
from pathlib import Path

class ContextExtensionTester:
    """Test context window extension on pretrained models."""
    
    def __init__(self, model_name="gpt2"):
        """Initialize with specified model."""
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.baseline_context_size = 1024  # GPT-2 training context
        print(f"Model loaded. Baseline context size: {self.baseline_context_size}")
    
    def generate_test_text(self, length):
        """Generate test text of specified token length."""
        # Use a repetitive but coherent text pattern
        base_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test sentence for context extension experiments. "
            "We need to understand how models handle longer sequences. "
            "Artificial intelligence and machine learning continue to evolve. "
        )
        
        # Repeat and truncate to desired length
        repeated_text = (base_text * (length // len(base_text.split()) + 1))
        tokens = self.tokenizer.encode(repeated_text)
        return self.tokenizer.decode(tokens[:length])
    
    def calculate_perplexity(self, text):
        """Calculate perplexity for given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            
        return perplexity
    
    def test_context_extension(self, max_context=2048, step_size=128):
        """Test context extension from baseline to max_context."""
        results = []
        context_sizes = list(range(self.baseline_context_size, max_context + 1, step_size))
        
        print(f"Testing context sizes: {context_sizes}")
        
        for context_size in tqdm(context_sizes, desc="Testing context sizes"):
            try:
                # Generate test text
                test_text = self.generate_test_text(context_size)
                
                # Measure perplexity
                start_time = time.time()
                perplexity = self.calculate_perplexity(test_text)
                inference_time = time.time() - start_time
                
                # Check for memory usage
                memory_used = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                result = {
                    'context_size': context_size,
                    'perplexity': perplexity,
                    'inference_time': inference_time,
                    'memory_used': memory_used,
                    'success': True,
                    'error': None
                }
                
                print(f"Context {context_size}: Perplexity={perplexity:.2f}, Time={inference_time:.2f}s")
                
            except Exception as e:
                result = {
                    'context_size': context_size,
                    'perplexity': None,
                    'inference_time': None,
                    'memory_used': None,
                    'success': False,
                    'error': str(e)
                }
                print(f"Context {context_size}: FAILED - {e}")
            
            results.append(result)
            
            # Early termination if we hit catastrophic failure
            if not result['success']:
                print("Stopping due to failure")
                break
        
        return results
    
    def analyze_results(self, results):
        """Analyze and visualize results."""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("No successful tests to analyze")
            return
        
        context_sizes = [r['context_size'] for r in successful_results]
        perplexities = [r['perplexity'] for r in successful_results]
        inference_times = [r['inference_time'] for r in successful_results]
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Perplexity vs Context Size
        ax1.plot(context_sizes, perplexities, 'b-o')
        ax1.axvline(x=self.baseline_context_size, color='r', linestyle='--', 
                   label=f'Training Context ({self.baseline_context_size})')
        ax1.set_xlabel('Context Size (tokens)')
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Perplexity vs Context Size')
        ax1.legend()
        ax1.grid(True)
        
        # Inference Time vs Context Size
        ax2.plot(context_sizes, inference_times, 'g-o')
        ax2.axvline(x=self.baseline_context_size, color='r', linestyle='--',
                   label=f'Training Context ({self.baseline_context_size})')
        ax2.set_xlabel('Context Size (tokens)')
        ax2.set_ylabel('Inference Time (seconds)')
        ax2.set_title('Inference Time vs Context Size')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('context_extension_results.png', dpi=300, bbox_inches='tight')
        print("Results saved to context_extension_results.png")
        
        # Analysis summary
        baseline_perplexity = perplexities[0]
        max_perplexity = max(perplexities)
        degradation_factor = max_perplexity / baseline_perplexity
        
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Baseline context size: {self.baseline_context_size}")
        print(f"Maximum tested context: {max(context_sizes)}")
        print(f"Baseline perplexity: {baseline_perplexity:.2f}")
        print(f"Maximum perplexity: {max_perplexity:.2f}")
        print(f"Degradation factor: {degradation_factor:.2f}x")
        
        if degradation_factor < 2.0:
            print("✅ RESULT: Graceful degradation - context extension viable!")
        elif degradation_factor < 5.0:
            print("⚠️  RESULT: Moderate degradation - context extension possible with fine-tuning")
        else:
            print("❌ RESULT: Severe degradation - context extension requires significant retraining")
        
        return {
            'degradation_factor': degradation_factor,
            'max_context_tested': max(context_sizes),
            'viable': degradation_factor < 2.0
        }
    
    def save_results(self, results, analysis):
        """Save results to JSON file."""
        output = {
            'experiment_info': {
                'model': 'gpt2',
                'baseline_context': self.baseline_context_size,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'raw_results': results,
            'analysis': analysis
        }
        
        with open('experiment_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("Detailed results saved to experiment_results.json")

def main():
    """Run the context extension experiment."""
    print("Context Window Extension Experiment")
    print("Testing GPT-2 context scaling without retraining")
    print("="*50)
    
    # Initialize tester
    tester = ContextExtensionTester()
    
    # Run experiment
    results = tester.test_context_extension(max_context=2048, step_size=128)
    
    # Analyze results
    analysis = tester.analyze_results(results)
    
    # Save everything
    tester.save_results(results, analysis)
    
    print("\n🎯 Experiment complete!")
    print("This data is crucial for understanding context scaling viability.")

if __name__ == "__main__":
    main()
