import json
import os
import argparse
import sys
sys.path.append("..")
import utils
import re


def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from model evaluation results")
    parser.add_argument("results_file", help="Path to the JSON file containing evaluation results")
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        detailed_results = json.load(f)['detailed_results']
    
    # Recalculate is_correct by extracting answers from full_response
    for item in detailed_results:
        extracted_answer = utils.extract_xml_answer(item.get("full_response", ""))
        if extracted_answer:
            item["model_answer"] = extracted_answer
        
        # Compare extracted answer with ground truth
        ground_truth = item.get("ground_truth", "").strip()
        model_answer = item.get("model_answer", "").strip()
        
        # Clean up model answer (remove $ and other non-numeric characters if needed)
        if utils.compare_str_numbers(model_answer,ground_truth):
            item["is_correct"] = True
        else:
            item["is_correct"] = False
    
    # Create results dictionary in the format expected by utils functions
    results = {}
    
    # Calculate accuracy
    total = len(detailed_results)
    correct = sum(1 for item in detailed_results if item.get("is_correct", False))
    results['accuracy'] = (correct / total) * 100 if total > 0 else 0
    results['total_examples'] = total
    results['correct'] = correct
    results['detailed_results'] = detailed_results
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Total examples: {results['total_examples']}")
    print(f"Correct answers: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    
    # Analyze errors using existing utils function
    utils.analyze_errors(results)

    # Default output path based on input filename
    base_name = os.path.splitext(args.results_file)[0]
    output_file = f"{base_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {output_file}")

if __name__ == "__main__":
    main()


