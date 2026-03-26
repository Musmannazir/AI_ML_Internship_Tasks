"""
Task 5: Mental Health Support Chatbot (Fine-Tuned)

Build an empathetic mental health support chatbot using fine-tuned DistilGPT2.

Features:
- Fine-tune on EmpatheticDialogues dataset (Facebook AI)
- Supports CLI and optional Streamlit interface
- Generates supportive responses for mental wellness queries
- Safety filters for urgent mental health situations

Training:
    python task5_mental_health_chatbot.py --train

Interactive Chat:
    python task5_mental_health_chatbot.py --chat

Requirements:
    pip install transformers datasets torch accelerate
    pip install streamlit  # Optional, for web interface
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# Configuration
MODEL_NAME = "distilgpt2"
OUTPUT_DIR = Path(__file__).parent / "fine_tuned_model"
CACHE_DIR = Path(__file__).parent / "model_cache"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# Safety patterns for mental health
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "hurt myself", "self harm",
    "want to die", "end my life", "overdose", "cut myself"
]

SUPPORT_PROMPTS = [
    "I'm feeling stressed",
    "I'm anxious about",
    "I'm sad about",
    "I'm overwhelmed by",
    "I'm struggling with",
    "I need help with",
    "I don't know how to cope",
    "Everything feels too much",
]

EMPATHETIC_SYSTEM = (
    "You are a compassionate mental health support chatbot. "
    "Respond with empathy, validation, and gentle guidance. "
    "Encourage the person to seek professional help when needed. "
    "Use warm, understanding language. "
    "Be supportive without providing medical advice."
)


def check_gpu_availability():
    """Check if GPU is available for training."""
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("⚠ No GPU detected. Training will be slow on CPU.")
        return False


def load_empathetic_dialogues() -> Any:
    """Load EmpatheticDialogues dataset from Hugging Face Hub."""
    print("Loading EmpatheticDialogues dataset...")
    try:
        from datasets import load_dataset

        dataset = load_dataset("empathetic_dialogues", split="train")
        print(f"✓ Loaded {len(dataset)} conversations")
        return dataset
    except ModuleNotFoundError:
        raise RuntimeError(
            "Missing dependency: 'datasets'. Install it with: pip install datasets"
        )
    except Exception as e:
        print(f"⚠ Could not load EmpatheticDialogues: {e}")
        print("Creating synthetic supportive conversation dataset...")
        return create_synthetic_dataset()


def create_synthetic_dataset() -> Any:
    """Create a synthetic supportive conversation dataset as fallback."""
    from datasets import Dataset

    data = {
        "utterance": [
            "I'm feeling really stressed about work lately.",
            "I understand. Work stress can be overwhelming.",
            "Thank you for listening. It helps to talk about it.",
            
            "I've been having trouble sleeping.",
            "Sleep issues can add to anxiety. Have you tried any relaxation techniques?",
            
            "I feel anxious all the time.",
            "Those feelings are valid. It might help to explore what triggers them.",
            
            "I don't know how to handle my emotions.",
            "That's a common struggle. Consider speaking with a therapist or counselor.",
            
            "Everything feels too much right now.",
            "I hear you. It's okay to feel overwhelmed. Take things one step at a time.",
            
            "I appreciate your support.",
            "You're welcome. Remember, seeking help is a sign of strength, not weakness.",
        ]
    }
    return Dataset.from_dict(data)


def prepare_dataset_for_finetuning(dataset: Any) -> Any:
    """Prepare dataset by concatenating utterances into conversations."""
    from datasets import Dataset

    print("Preparing dataset for fine-tuning...")
    
    # Group by conversation and create training examples
    texts = []
    
    if "utterance" in dataset.column_names:
        # EmpatheticDialogues format
        for utterance in dataset["utterance"]:
            if utterance and len(str(utterance)) > 5:
                texts.append(str(utterance))
    elif "conv_id" in dataset.column_names:
        # Alternative format with conversations
        current_conv = []
        current_id = None
        
        for item in dataset:
            if item["conv_id"] != current_id:
                if current_conv:
                    texts.append(" ".join(current_conv))
                current_conv = [item["utterance"]]
                current_id = item["conv_id"]
            else:
                current_conv.append(item["utterance"])
        
        if current_conv:
            texts.append(" ".join(current_conv))
    
    if not texts:
        # Fallback: use dataset as is
        texts = [str(item) for item in dataset]
    
    dataset = Dataset.from_dict({"text": texts})
    print(f"✓ Prepared {len(dataset)} training examples")
    return dataset


def tokenize_function(examples, tokenizer):
    """Tokenize examples for training."""
    # Defensive check: some GPT tokenizers do not define a pad token by default.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def train_model(dataset: Optional[Any] = None):
    """Fine-tune model on empathetic dialogue dataset."""
    check_gpu_availability()
    
    print("\n" + "="*60)
    print("Starting Model Fine-Tuning...")
    print("="*60)
    
    # Load model and tokenizer
    print(f"\nLoading {MODEL_NAME} model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Set padding token for GPT-like tokenizers.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    
    # Load dataset
    if dataset is None:
        dataset = load_empathetic_dialogues()
        dataset = prepare_dataset_for_finetuning(dataset)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=100,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    print("\nTraining model...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("✓ Model saved successfully")


def load_finetuned_model():
    """Load fine-tuned model from disk."""
    if not OUTPUT_DIR.exists():
        print(f"✗ Model not found at {OUTPUT_DIR}")
        print("Train the model first: python task5_mental_health_chatbot.py --train")
        return None, None
    
    print(f"Loading fine-tuned model from {OUTPUT_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(OUTPUT_DIR))
    return model, tokenizer


def check_crisis_keywords(text: str) -> str | None:
    """Check for crisis keywords and return safety response if found."""
    text_lower = text.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return (
                "I'm concerned about what you've shared. If you're experiencing thoughts of self-harm "
                "or suicide, please reach out for immediate help:\n\n"
                "National Suicide Prevention Lifeline: 988 (US)\n"
                "Crisis Text Line: Text HOME to 741741\n"
                "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/\n\n"
                "You deserve support, and there are people trained to help."
            )
    return None


def generate_response(model, tokenizer, user_input: str, max_length: int = 100) -> str:
    """Generate empathetic response using fine-tuned model."""
    # Check for crisis keywords
    crisis_response = check_crisis_keywords(user_input)
    if crisis_response:
        return crisis_response
    
    # Format input with empathetic context
    prompt = f"User: {user_input}\nAssistant:"
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=3,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response or "I'm here to listen and support you. Please tell me more."


def run_cli_chat():
    """Run interactive CLI chat."""
    model, tokenizer = load_finetuned_model()
    if model is None:
        return
    
    print("\n" + "="*60)
    print("Mental Health Support Chatbot")
    print("="*60)
    print("Type 'exit' to quit.\n")
    print("Note: This chatbot provides emotional support but is not a substitute")
    print("for professional mental health services.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("\nBot: Thank you for sharing with me. Remember to be kind to yourself. Take care!")
            break
        
        if not user_input:
            print("Bot: I'm here to listen. Please share what's on your mind.\n")
            continue
        
        try:
            response = generate_response(model, tokenizer, user_input)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Bot: I encountered an issue: {e}\n")


def run_streamlit_app():
    """Run Streamlit web interface (optional)."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Install with: pip install streamlit")
        return
    
    st.set_page_config(page_title="Mental Health Support", layout="centered")
    st.title("🌟 Mental Health Support Chatbot")
    
    st.markdown("""
    This chatbot provides emotional support and empathetic responses
    for stress, anxiety, and emotional wellness.
    
    **Note:** This is not a substitute for professional mental health care.
    """)
    
    # Load model
    model, tokenizer = load_finetuned_model()
    if model is None:
        st.error("Model not found. Please train the model first.")
        return
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    if user_input := st.chat_input("Share what's on your mind..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.spinner("Listening..."):
            response = generate_response(model, tokenizer, user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.write(response)
    
    # Crisis resources
    with st.expander("🆘 Crisis Resources"):
        st.markdown("""
        If you're in immediate danger or experiencing a mental health crisis:
        
        **US:**
        - National Suicide Prevention Lifeline: 988
        - Crisis Text Line: Text HOME to 741741
        
        **International:**
        - [IASP Crisis Centers](https://www.iasp.info/resources/Crisis_Centres/)
        """)


def is_running_under_streamlit() -> bool:
    """Detect whether this script is currently running via Streamlit."""
    if "streamlit" in sys.modules:
        return True

    if any("streamlit" in arg.lower() for arg in sys.argv):
        return True

    if os.getenv("STREAMLIT_SERVER_PORT"):
        return True

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mental Health Support Chatbot with Fine-Tuning"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fine-tune the model on empathetic dialogues",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Run interactive CLI chat (default)",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Run Streamlit web interface",
    )
    
    args, _unknown = parser.parse_known_args()

    # When launched with `streamlit run`, force web app mode.
    if is_running_under_streamlit():
        args.streamlit = True
        args.chat = False
        args.train = False
    
    # If no specific action, choose mode based on environment.
    if not args.train and not args.chat and not args.streamlit:
        # Streamlit runs without an interactive stdin; CLI chat requires it.
        if not sys.stdin.isatty():
            args.streamlit = True
        else:
            args.chat = True
    
    try:
        if args.train:
            train_model()
            return

        # In non-interactive environments (e.g., streamlit runner), never enter CLI input loop.
        if args.streamlit or not sys.stdin.isatty():
            run_streamlit_app()
        else:  # args.chat
            run_cli_chat()
    except KeyboardInterrupt:
        print("\n\nExiting... Take care of yourself!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
