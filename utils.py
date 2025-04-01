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
""",
    },
    "v1": {
        "SYSTEM_PROMPT": """Let's think step by step.
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
""",
    },
    "v2": {
        "SYSTEM_PROMPT": """First, understand the problem and analyze any of the provided demonstrations if they're relevant. Then, solve the problem step by step.
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
""",
    },
    "v3": {
        "SYSTEM_PROMPT": """First, understand the problem and analyze any of the provided demonstrations if they're relevant. Then, solve the problem step by step.""",
        "XML_COT_FORMAT": """\
{reasoning}
####
{answer}
""",
    }
}

# Default to the latest version for backward compatibility


def extract_xml_answer(text: str) -> str:
    # Try to extract from XML tags
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
    else:
        # If no answer tags, take the last line
        answer = text.strip()
    
    # Simple pattern to match any number with optional commas and decimal points
    number_matches = list(re.finditer(r'\d+(?:,\d+)*(?:\.\d+)?', answer))
    if number_matches:
        # Get the last number found and remove any commas
        return number_matches[-1].group(0).replace(',', '')
    
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def compare_str_numbers(x,y):
    # Try to convert both to float for numerical comparison
    model_float = float(x.replace(',', ''))
    truth_float = float(y.replace(',', ''))
    is_correct = abs(model_float - truth_float) < 1e-8  # Account for floating point precision
    return is_correct 

def get_gsm8k_questions(split = "train", prompt_version = "v0", few_shot=False, k_shot=4, few_shot_template="chat") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    SYSTEM_PROMPT = PROMPTS[prompt_version]["SYSTEM_PROMPT"]
    XML_COT_FORMAT = PROMPTS[prompt_version]["XML_COT_FORMAT"]
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
            few_shot_prompt += f"Now, answer this problem.\n\nQuestion: "
            
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
def get_dataset(dataset_name, split="train", prompt_version = "v0", few_shot=False, k_shot=4, few_shot_template="chat"):
    if dataset_name == "gsm8k":
        return get_gsm8k_questions(split, prompt_version, few_shot, k_shot, few_shot_template)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

# Evaluate on GSM8K test set
def evaluate_model(model, test_data, tokenizer, lora_path=None):
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
            item['prompt'],
            tokenize=False, 
            add_generation_prompt=True
        )
        # Generate response
        output = model.fast_generate(
            prompt,
            sampling_params=sampling_params,
            lora_request=lora_request
        )[0].outputs[0].text
        
        # Extract and compare answer
        try:
            model_answer = extract_xml_answer(output)
            ground_truth = item['answer']
            # Convert to float and compare numerically if both are numeric
            try:
                is_correct = compare_str_numbers(model_answer,ground_truth)
            except (ValueError, TypeError):
                # If conversion fails, fall back to string comparison
                is_correct = model_answer == ground_truth
            if is_correct:
                correct += 1
            
            # Save detailed results
            results.append({
                'question': item['question'],
                'prompt': prompt,  # The last message is the question
                'ground_truth': ground_truth,
                'model_answer': model_answer,
                'full_response': output,
                'is_correct': is_correct
            })
            
        except Exception as e:
            print(f"Error processing answer: {e}")
            results.append({
                'question': item['question'],
                'prompt': prompt,  
                'ground_truth': ground_truth,
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
