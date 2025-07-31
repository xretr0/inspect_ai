#!/usr/bin/env python3
"""
Test script for the NNSight provider with Inspect AI.

This script demonstrates how to:
1. Create a nnsight LanguageModel instance
2. Pass it to the Inspect AI get_model function via model_args
3. Create a simple evaluation task
4. Run eval with the nnsight provider
"""

import torch
from nnsight import LanguageModel
from inspect_ai import Task, eval, task
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


@task
def nnsight_test_task():
    """Simple test task for nnsight provider."""
    return Task(
        dataset=[
            Sample(
                input="Complete the sequence: A B C D E F G H I J K L M N",
                target="O",  # Next letter in alphabet
            ),
            Sample(
                input="What comes after 1, 2, 3, 4, 5?",
                target="6",
            ),
            Sample(
                input="Say hello",
                target="Hello",  # Simple greeting test
            ),
        ],
        solver=[
            generate(),
        ],
        scorer=exact(),
    )



class NNSightModel1:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.activations_cache = {}
        self.model = LanguageModel(
            self.model_name, 
            device_map="auto")

    def generate(self, input: str, config: GenerateConfig) -> str:
        input_token_length = len(self.model.tokenizer.encode(input))
        with self.model.generate(input, max_new_tokens=config.max_tokens):
            self.activations_cache["layer_5"] = self.model.model.layers[5].mlp.output[0].save()
            output = self.model.generator.output.save()
        
        return self.model.tokenizer.decode(output[0][input_token_length :].cpu())
    
    def print_activations(self):
        print(self.activations_cache["layer_5"])


class NNSightModel2:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = LanguageModel(
            self.model_name, 
            device_map="auto")
        self.activations_cache = {}
        self.edit()
    
    def edit(self) -> None:
        with self.model.edit(inplace=True):
            self.model.model.layers[5].mlp.output[0][:] = torch.randn_like(
                self.model.model.layers[5].mlp.output[0]
            )
    
    def print_activations(self):
        print(self.activations_cache["layer_5"])

    def hook(self, input: str) -> None:
        self.activations_cache["layer_5"] = self.model.model.layers[5].mlp.output[0].save()


def main():
    """Main function that creates nnsight model and runs evaluation."""
    # nnsight_model1 = NNSightModel1("meta-llama/Llama-3.1-8B-Instruct")
    
    # inspect_model1 = get_model(
    #     model="custom/my_model",
    #     generate_function=nnsight_model1.generate,
    # )
    
    # # Run the evaluation using the nnsight provider
    # # The model name format is "nnsight/<model_name>" and we pass the actual
    # # nnsight model instance via model_args
    # eval_logs = eval(
    #     nnsight_test_task(),
    #     model=inspect_model1,
    #     max_tokens=100,
    # )

    # print(eval_logs)
    # nnsight_model1.print_activations()

    nnsight_model2 = NNSightModel2("meta-llama/Llama-3.1-8B-Instruct")
    
    inspect_model2 = get_model(
        model="nnsight/my_model",
        nnsight_model=nnsight_model2.model,
        nnsight_hook=nnsight_model2.hook,
    )
    
    eval_logs = eval(
        nnsight_test_task(),
        model=inspect_model2,
        max_tokens=100,
    )

    print(eval_logs)
    nnsight_model2.print_activations()

if __name__ == "__main__":
    main()