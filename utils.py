from vllm import SamplingParams
from tqdm import tqdm
import re
from datasets import load_dataset, Dataset
import re

# Prompts
PROMPTS = {
    "v0": {
        "SYSTEM_PROMPT": """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
""",
        "XML_COT_FORMAT": """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""
    },
    "v1": {
        "SYSTEM_PROMPT": """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
""",
        "XML_COT_FORMAT": """\
<think>
{reasoning}
</think>
<answer>
{answer}
</answer>
"""
    }
}

# Default to the latest version for backward compatibility
SYSTEM_PROMPT = PROMPTS["v0"]["SYSTEM_PROMPT"]
XML_COT_FORMAT = PROMPTS["v0"]["XML_COT_FORMAT"]

def extract_xml_answer(text: str) -> str:
    # Try to extract from XML tags
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
    else:
        # If no answer tags, take the last line
        answer = text.strip().split("\n")[-1]
    
    # Extract the last number in the answer using regex that handles commas in numbers
    number_matches = list(re.finditer(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?', answer.strip()))
    if number_matches:
        # Get the last match
        last_match = number_matches[-1]
        # Remove commas from the number
        return last_match.group(0).replace(',', '')
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "train", few_shot=False, k_shot=5, few_shot_template="chat") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    
    if few_shot:
        # Get k random examples from training set for few-shot learning
        train_data = load_dataset('openai/gsm8k', 'main')["train"] # type: ignore
        # Sample k random examples
        few_shot_examples = train_data.shuffle(seed=42).select(range(k_shot)) # type: ignore
        
        if few_shot_template == "combined":
            # Create a combined few-shot prompt with all examples concatenated
            few_shot_prompt = ""
            for i, example in enumerate(few_shot_examples):
                few_shot_prompt += f"### Example {i+1}\n"
                few_shot_prompt += f"Question: {example['question']}\n\n"
                few_shot_prompt += XML_COT_FORMAT.format(
                    reasoning=example['answer'].split('####')[0].strip(),
                    answer=extract_hash_answer(example['answer'])
                )
                few_shot_prompt += "\n\n"
            
            # Add separator before the actual question
            few_shot_prompt += "Now, solve the following problem:\n\nQuestion: "
            
            # Process the data with combined few-shot examples
            data = data.map(lambda x: { # type: ignore
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': few_shot_prompt + x['question']}
                ],
                'answer': extract_hash_answer(x['answer'])
            }) # type: ignore
        else:  # Default to chat format
            # Process the data with few-shot examples in chat format
            data = data.map(lambda x: { # type: ignore
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    # Include few-shot examples in the prompt
                    *sum([[
                        {'role': 'user', 'content': example['question']},
                        {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                            reasoning=example['answer'].split('####')[0].strip(),
                            answer=extract_hash_answer(example['answer'])
                        )}
                    ] for example in few_shot_examples], []),
                    # Add the actual question
                    {'role': 'user', 'content': x['question']}
                ],
                'answer': extract_hash_answer(x['answer'])
            }) # type: ignore
    else:
        # Standard processing without few-shot examples
        data = data.map(lambda x: { # type: ignore
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        }) # type: ignore
    
    return data # type: ignore

# Function to get dataset based on argument
def get_dataset(dataset_name, split="train", few_shot=False, k_shot=5):
    if dataset_name == "gsm8k":
        return get_gsm8k_questions(split, few_shot, k_shot)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
# Reward functions
def correctness_reward_func(prompts, completions, answer, debug=True, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    if debug:
        print(prompts)
        print(completions)
        print(answer)
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# Evaluate on GSM8K test set
def evaluate_model(model, test_data, tokenizer, lora_path=None, few_shot=False, k_shot=5):
    correct = 0
    total = 0
    results = []
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    
    # Load LoRA weights if provided
    lora_request = None
    if lora_path:
        lora_request = model.load_lora(lora_path)
    
    # Process each question in the test set
    for item in tqdm(test_data):
        # The prompt is already formatted in the dataset
        prompt = tokenizer.apply_chat_template(
            item['prompt'][0],
            tokenize=False, 
            add_generation_prompt=True
        )
        # Generate response
        output = model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=lora_request
        )[0].outputs[0].text
        
        # Extract and compare answer
        try:
            model_answer = extract_xml_answer(output)
            ground_truth = item['answer']
            
            is_correct = model_answer == ground_truth
            if is_correct:
                correct += 1
            
            # Save detailed results
            results.append({
                'question': item['question'],
                'prompt': item['prompt'][0],  # The last message is the question
                'ground_truth': ground_truth,
                'model_answer': model_answer,
                'full_response': output,
                'is_correct': is_correct
            })
            
        except Exception as e:
            print(f"Error processing answer: {e}")
            results.append({
                'question': item['question'],
                'prompt': item['prompt'][0],  
                'ground_truth': item['answer'],
                'model_answer': "ERROR",
                'full_response': output,
                'is_correct': False
            })
        
        total += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"GSM8K Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'detailed_results': results
    }

def analyze_errors(results):
    errors = [r for r in results['detailed_results'] if not r['is_correct']]
    
    # Check for common error patterns
    numeric_but_wrong = 0
    format_errors = 0
    
    for error in errors:
        model_answer = error['model_answer']
        if model_answer.isdigit() or (model_answer.replace('.', '', 1).isdigit() and model_answer.count('.') <= 1):
            numeric_but_wrong += 1
        if "<answer>" not in error['full_response'] or "<reasoning>" not in error['full_response']:
            format_errors += 1
    
    print("\nError Analysis:")
    print(f"Total errors: {len(errors)}")
    print(f"Numeric but wrong: {numeric_but_wrong} ({numeric_but_wrong/len(errors)*100:.1f}%)")
    print(f"Format errors: {format_errors} ({format_errors/len(errors)*100:.1f}%)")
