#!/usr/bin/env python3
"""Test script for transformer_lens integration with inspect_ai."""

import asyncio

from transformer_lens import HookedTransformer

# Import from the local modified inspect_ai
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model


async def test_transformer_lens_integration():
    """Test the transformer_lens provider integration."""
    print("1. Loading HookedTransformer model...")
    # Create a HookedTransformer instance - using a small model for testing
    model = HookedTransformer.from_pretrained(
        "gpt2",  # Using small GPT-2 for testing
        device="cuda",  # Use CPU for testing
    )
    print(f"   ✓ Model loaded: {model.cfg.model_name}")

    print("\n2. Creating inspect_ai model with transformer_lens provider...")
    # Create model_args for the transformer_lens provider
    model_args = {
        "tl_model": model,  # Pass the HookedTransformer instance (renamed to avoid conflict)
        "tl_generate_args": {  # Generation arguments
            "max_new_tokens": 50,
            "temperature": 0.7,
            "do_sample": True,
        },
    }

    # Get the model using inspect_ai with our custom provider
    inspect_model = get_model(
        "transformer_lens/gpt2",  # provider/model_name format
        **model_args,  # Now safe to unpack since we renamed 'model' to 'tl_model'
    )
    print("   ✓ Inspect AI model created")

    print("\n3. Testing generation...")
    # Create a test message
    messages = [ChatMessageUser(content="a b c d e f g h i j")]

    # Generate a response
    try:
        response = await inspect_model.generate(
            input=messages,
            tools=[],  # No tools for this test
            tool_choice="none",
            config=GenerateConfig(max_tokens=50),
        )

        print("   ✓ Generation successful!")
        print(f"\n4. Response:\n   {response.choices[0].message.content}")

        # Print some metadata if available
        if response.usage:
            print("\n5. Token usage:")
            print(f"   - Input tokens: {response.usage.input_tokens}")
            print(f"   - Output tokens: {response.usage.output_tokens}")
            print(f"   - Total tokens: {response.usage.total_tokens}")

    except Exception as e:
        print(f"   ✗ Generation failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print("\n✅ Test completed!")


if __name__ == "__main__":
    # Run the async test function
    asyncio.run(test_transformer_lens_integration())
