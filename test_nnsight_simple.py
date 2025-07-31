#!/usr/bin/env python3
"""
Simple test script for NNSight provider with Inspect AI.

This script demonstrates the basic workflow:
1. Create nnsight model
2. Pass to get_model 
3. Call eval function
"""

from nnsight import LanguageModel
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


@task
def simple_test():
    """Simple test task."""
    return Task(
        dataset=[
            Sample(
                input="A B C D E F G H I J K L M N O",
                target="P",
            )
        ],
        solver=[generate()],
        scorer=exact(),
    )


def main():
    # Create the nnsight model (like your example)
    test_model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
    print(f"Created model: {test_model}")
    
    # Run eval with nnsight provider, passing the model instance
    result = eval(
        simple_test(),
        model="nnsight/meta-llama/Llama-3.1-8B-Instruct",
        model_args={"nnsight_model": test_model}
    )
    
    print(f"Evaluation completed: {result[0].status}")


if __name__ == "__main__":
    main()