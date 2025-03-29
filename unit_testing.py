import utils

def test_extract_xml_answer():
    test_cases = [
        # Basic XML format tests
        {
            "input": "<reasoning>\nSome reasoning\n</reasoning>\n<answer>\n42\n</answer>",
            "expected": "42",
            "name": "Simple number"
        },
        {
            "input": "<reasoning>\nCalculation\n</reasoning>\n<answer>\nThe answer is 123,456\n</answer>",
            "expected": "123456",
            "name": "Number with commas"
        },
        {
            "input": "<reasoning>\nMath\n</reasoning>\n<answer>\nJessica pays $18000 in a year.\n</answer>",
            "expected": "18000",
            "name": "Number in sentence"
        },
        {
            "input": "<reasoning>\nDecimals\n</reasoning>\n<answer>\n42.5\n</answer>",
            "expected": "42.5",
            "name": "Decimal number"
        },
        # Edge cases
        {
            "input": "<answer>100</answer>",
            "expected": "100",
            "name": "No reasoning tag"
        },
        {
            "input": "Just a number 42",
            "expected": "42",
            "name": "No XML tags"
        },
        {
            "input": "<reasoning>\nMultiple numbers 10 20 30\n</reasoning>\n<answer>\n30\n</answer>",
            "expected": "30",
            "name": "Multiple numbers"
        },
        {
            "input": "<answer>No numbers here</answer>",
            "expected": "No numbers here",
            "name": "No numbers"
        },
        # Complex cases
        {
            "input": "<reasoning>\nComplex calculation\n</reasoning>\n<answer>\nThe final result is $1,234,567.89\n</answer>",
            "expected": "1234567.89",
            "name": "Complex number format"
        },
        {
            "input": "<reasoning>\nYear calculation\n</reasoning>\n<answer>\nIn 2023, the value is 500\n</answer>",
            "expected": "500",
            "name": "Number with year"
        },
                # LaTeX format tests
        {
            "input": "<reasoning>\nCalculation\n</reasoning>\n<answer>\n\\[\n\\boxed{539}\n\\]\n</answer>",
            "expected": "539",
            "name": "Simple LaTeX boxed"
        },
        {
            "input": "<reasoning>\nMath\n</reasoning>\n<answer>\n\\[\n\\boxed{1,234.56}\n\\]\n</answer>",
            "expected": "1234.56",
            "name": "LaTeX boxed with comma and decimal"
        },
        {
            "input": "<reasoning>\nComplex\n</reasoning>\n<answer>\nThe answer is \\boxed{42} in LaTeX\n</answer>",
            "expected": "42",
            "name": "LaTeX boxed in sentence"
        },
        {
            "input": "<reasoning>\nMultiple\n</reasoning>\n<answer>\n\\[\n\\boxed{10} + \\boxed{20} = \\boxed{30}\n\\]\n</answer>",
            "expected": "30",
            "name": "Multiple LaTeX boxed numbers"
        },
        {
            "input": "<reasoning>\nBig number\n</reasoning>\n<answer>\n\\[\n\\boxed{1,000,000}\n\\]\n</answer>",
            "expected": "1000000",
            "name": "LaTeX boxed large number with commas"
        }
    ]
    
    failed_tests = []
    passed = 0
    
    for test in test_cases:
        try:
            result = utils.extract_xml_answer(test["input"])
            if result == test["expected"]:
                passed += 1
                print(f"✓ {test['name']}: PASSED")
            else:
                failed_tests.append({
                    "name": test["name"],
                    "input": test["input"],
                    "expected": test["expected"],
                    "got": result
                })
                print(f"✗ {test['name']}: FAILED")
        except Exception as e:
            failed_tests.append({
                "name": test["name"],
                "input": test["input"],
                "expected": test["expected"],
                "error": str(e)
            })
            print(f"✗ {test['name']}: ERROR - {str(e)}")
    
    print(f"\nTest Summary:")
    print(f"Passed: {passed}/{len(test_cases)} tests")
    
    if failed_tests:
        print("\nFailed Tests Details:")
        for fail in failed_tests:
            print(f"\n{fail['name']}:")
            print(f"Input: {fail['input']}")
            print(f"Expected: {fail['expected']}")
            if "got" in fail:
                print(f"Got: {fail['got']}")
            if "error" in fail:
                print(f"Error: {fail['error']}")

# Run the tests
if __name__ == "__main__":
    test_extract_xml_answer()