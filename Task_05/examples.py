"""
Example: Using Task 5 Model Programmatically

This script shows how to load and use the fine-tuned mental health chatbot
in your own Python application without using the CLI.
"""

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to fine-tuned model
MODEL_PATH = Path(__file__).parent / "fine_tuned_model"


def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train it first: python task5_mental_health_chatbot.py --train"
        )
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH))
    print("✓ Model loaded successfully")
    
    return model, tokenizer


def generate_response(model, tokenizer, user_message: str, max_length: int = 100) -> str:
    """
    Generate a supportive response to a user message.
    
    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        user_message: User's emotional query
        max_length: Maximum response length
    
    Returns:
        Supportive response from the chatbot
    """
    prompt = f"User: {user_message}\nAssistant:"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=3,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response


def example_usage():
    """Example of using the model in a batch processing scenario."""
    
    # Load model once
    model, tokenizer = load_model_and_tokenizer()
    
    # Example conversations
    user_messages = [
        "I'm feeling really stressed about my job.",
        "I've been having trouble sleeping lately.",
        "Everything feels overwhelming right now.",
        "I don't know how to handle my anxiety.",
    ]
    
    print("\nGenerating supportive responses:\n")
    print("=" * 70)
    
    for user_msg in user_messages:
        response = generate_response(model, tokenizer, user_msg)
        print(f"\nUser: {user_msg}")
        print(f"Bot: {response}")
        print("-" * 70)


def create_conversation_history(model, tokenizer, messages: list[str]) -> list[dict]:
    """
    Generate responses for a multi-turn conversation.
    
    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        messages: List of user messages
    
    Returns:
        Conversation history with user and bot messages
    """
    conversation = []
    
    for user_msg in messages:
        response = generate_response(model, tokenizer, user_msg)
        conversation.append({
            "role": "user",
            "content": user_msg,
        })
        conversation.append({
            "role": "assistant",
            "content": response,
        })
    
    return conversation


def export_conversation(conversation: list[dict], filename: str = "conversation.txt"):
    """Save conversation history to a file."""
    with open(filename, "w") as f:
        for msg in conversation:
            role = msg["role"].capitalize()
            f.write(f"{role}: {msg['content']}\n\n")
    
    print(f"Conversation saved to {filename}")


if __name__ == "__main__":
    print("Task 5 - Model Usage Examples\n")
    print("="*70)
    
    # Example 1: Single response generation
    print("\n--- Example 1: Single Response ---\n")
    model, tokenizer = load_model_and_tokenizer()
    
    user_input = "I've been feeling really down lately."
    response = generate_response(model, tokenizer, user_input)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")
    
    # Example 2: Batch processing
    print("--- Example 2: Batch Processing ---")
    print("="*70)
    example_usage()
    
    # Example 3: Multi-turn conversation
    print("\n--- Example 3: Multi-turn Conversation ---\n")
    print("="*70)
    messages = [
        "I'm feeling anxious about my upcoming presentation.",
        "Any tips for managing my nervousness?",
        "Thank you for the advice.",
    ]
    
    conversation = create_conversation_history(model, tokenizer, messages)
    
    print("\nConversation History:")
    print("-" * 70)
    for msg in conversation:
        role = msg["role"].capitalize()
        print(f"{role}: {msg['content']}\n")
    
    # Save to file
    export_conversation(conversation)
    
    print("\n" + "="*70)
    print("Examples complete!")
