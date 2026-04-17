"""
Task 4: General Health Query Chatbot (Prompt Engineering Based)

Features:
- Uses an LLM via API (OpenAI Chat Completions or Hugging Face Inference API)
- Applies prompt engineering for friendly, clear health education responses
- Adds safety filters to avoid harmful medical advice
- Provides an interactive terminal chat experience

Environment variables:
- OPENAI_API_KEY       (preferred backend if set)
- OPENAI_MODEL         (default: gpt-3.5-turbo)
- HF_API_KEY           (fallback backend)
- HF_MODEL             (default: mistralai/Mistral-7B-Instruct-v0.2)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal

import requests

Backend = Literal["openai", "huggingface", "local"]


SYSTEM_PROMPT = (
    "You are a helpful medical assistant for general health education. "
    "Use friendly, clear, non-technical language. "
    "You are not a doctor and must not provide diagnosis, prescriptions, "
    "specific medication dosages, or emergency treatment instructions. "
    "If user asks for risky or urgent advice, tell them to contact a licensed "
    "health professional or emergency services immediately. "
    "Keep answers concise and practical in 4 to 8 sentences."
)


EMERGENCY_PATTERNS = [
    r"chest pain",
    r"can[']?t breathe|shortness of breath",
    r"stroke",
    r"seizure",
    r"faint(ed|ing)?",
    r"suicid(e|al)",
    r"overdose",
    r"severe bleeding",
]

HARMFUL_INTENT_PATTERNS = [
    r"how to (harm|hurt|kill) myself",
    r"how to overdose",
    r"what is a lethal dose",
    r"how to poison",
    r"stop (my|the) medicine immediately",
    r"replace doctor advice",
]

DOSAGE_PATTERN = re.compile(r"\b\d+\s?(mg|mcg|g|ml|tablets?|pills?)\b", re.IGNORECASE)


@dataclass
class HealthChatbot:
    backend: Backend
    api_key: str
    model: str
    history: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_environment(cls) -> "HealthChatbot":
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        hf_key = os.getenv("HF_API_KEY", "").strip()

        # Check if OpenAI key is valid (must start with sk-)
        if openai_key and openai_key.startswith("sk-"):
            return cls(
                backend="openai",
                api_key=openai_key,
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo").strip() or "gpt-3.5-turbo",
            )

        if hf_key:
            return cls(
                backend="huggingface",
                api_key=hf_key,
                model=os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2").strip()
                or "mistralai/Mistral-7B-Instruct-v0.2",
            )

        # Fallback to local mode (no API needed)
        return cls(
            backend="local",
            api_key="",
            model="local",
        )

    def answer(self, user_query: str) -> str:
        user_query = user_query.strip()
        if not user_query:
            return "Please ask a health-related question, and I will try to help."

        # Hard safety gate before model call.
        safety_block = self._safety_precheck(user_query)
        if safety_block:
            return safety_block

        if self.backend == "openai":
            response_text = self._ask_openai(user_query)
        elif self.backend == "huggingface":
            response_text = self._ask_huggingface(user_query)
        else:
            response_text = self._ask_local(user_query)

        safe_response = self._safety_postcheck(response_text)

        # Keep short chat memory for context without growing prompt indefinitely.
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": safe_response})
        self.history = self.history[-8:]

        return safe_response

    def _safety_precheck(self, user_query: str) -> str | None:
        text = user_query.lower()

        if any(re.search(pattern, text) for pattern in HARMFUL_INTENT_PATTERNS):
            return (
                "I cannot help with harmful or dangerous requests. "
                "If you are in immediate danger or thinking about self-harm, "
                "please contact local emergency services right now."
            )

        if any(re.search(pattern, text) for pattern in EMERGENCY_PATTERNS):
            return (
                "This could be urgent. Please seek immediate medical care or call emergency services now. "
                "I can only provide general information, not emergency guidance."
            )

        return None

    def _safety_postcheck(self, model_response: str) -> str:
        text = model_response.strip()

        # If generated text includes specific dosing-like content, replace with safer guidance.
        if DOSAGE_PATTERN.search(text):
            return (
                "I can share general health information, but I cannot provide medication doses. "
                "Please ask a licensed doctor or pharmacist for safe dosage guidance, "
                "especially for children, pregnancy, older adults, or existing conditions."
            )

        return text

    def _ask_openai(self, user_query: str) -> str:
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history[-6:])
        messages.append({"role": "user", "content": user_query})

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 300,
            },
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"].strip()
        return content or "I could not generate a response. Please try again."

    def _ask_huggingface(self, user_query: str) -> str:
        prompt = self._build_instruction_prompt(user_query)
        endpoint = f"https://api-inference.huggingface.co/models/{self.model}"

        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 280,
                        "temperature": 0.3,
                        "return_full_text": False,
                    },
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            # Hugging Face inference output formats can vary by model pipeline.
            if isinstance(data, list) and data and "generated_text" in data[0]:
                text = str(data[0]["generated_text"]).strip()
            elif isinstance(data, dict) and "generated_text" in data:
                text = str(data["generated_text"]).strip()
            else:
                text = ""

            return text or "I could not generate a response. Please try again."
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 410:
                raise RuntimeError(
                    f"The Hugging Face model '{self.model}' is no longer available on the free tier. "
                    "Please set OPENAI_API_KEY environment variable to use OpenAI API instead, "
                    "or try a different HF_MODEL (e.g., 'gpt2' or 'google/flan-t5-base')."
                ) from e
            raise

    def _ask_local(self, user_query: str) -> str:
        """Local rule-based response generator (no API required)."""
        query_lower = user_query.lower()
        
        # Knowledge base for common health queries
        knowledge_base = {
            "sore throat": "A sore throat is often caused by viral infections like the common cold or flu. It can also result from bacterial infections, dry air, or irritants. Most cases improve with rest, fluids, and throat lozenges. If it persists beyond a week or is very painful, see a healthcare provider.",
            "paracetamol|acetaminophen|tylenol": "Paracetamol can be used in children, but dosage depends on age and weight. Always follow the packaging instructions or consult a pharmacist. Never exceed recommended doses. For specific dosing for children, ask a doctor or pharmacist.",
            "panadol": "Panadol (paracetamol) can be safe for children when used correctly. The dose depends on the child's age and weight. Always read the label and follow dosing instructions. For children under 2 years, consult a healthcare provider first.",
            "fever": "Fever is often the body's way of fighting infection. Keep the person hydrated and comfortable. Over-the-counter pain relievers may help. Seek medical care if fever is very high, lasts more than 3 days, or is accompanied by other concerning symptoms.",
            "cold|cough": "Common colds are viral and typically resolve on their own in 7-10 days. Rest, fluids, and honey can help soothe symptoms. Most cough remedies target symptom relief rather than cure. If symptoms persist or worsen, see a healthcare provider.",
            "headache": "Headaches have many causes: stress, dehydration, sleep issues, or illness. Try rest, hydration, and a dark quiet room. Over-the-counter pain relievers may help. If headaches are severe, frequent, or unusual, consult a doctor.",
            "safe|children|kids|child": "Medication safety for children is important. Always check age-appropriate dosing on packaging. Never give adult medications without consulting a pharmacist or doctor. When in doubt, contact your local poison control or healthcare provider.",
        }
        
        # Find matching responses
        for keyword, response in knowledge_base.items():
            if re.search(keyword, query_lower):
                return response
        
        # Default response for unknown queries
        return (
            "I can provide general health information, but I'm limited without real-time data. "
            "For your specific question, I recommend consulting a healthcare professional or pharmacist who can provide personalized advice based on your situation."
        )

    def _build_instruction_prompt(self, user_query: str) -> str:
        conversation = []
        for msg in self.history[-6:]:
            if msg["role"] == "user":
                conversation.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                conversation.append(f"Assistant: {msg['content']}")

        conversation_text = "\n".join(conversation)
        if conversation_text:
            conversation_text += "\n"

        return (
            f"System: {SYSTEM_PROMPT}\n"
            f"{conversation_text}"
            f"User: {user_query}\n"
            "Assistant:"
        )


def run_chat() -> None:
    print("General Health Query Chatbot")
    print("Type 'exit' to quit.\n")
    print("Example queries:")
    print("- What causes a sore throat?")
    print("- Is paracetamol safe for children?\n")

    try:
        bot = HealthChatbot.from_environment()
    except RuntimeError as exc:
        print(f"Setup error: {exc}")
        print("Tip: set OPENAI_API_KEY (or HF_API_KEY) and run again.")
        return

    print(f"Using backend: {bot.backend} | model: {bot.model}\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"exit", "quit", "bye"}:
            print("Bot: Take care. Goodbye!")
            break

        try:
            answer = bot.answer(user_query)
        except RuntimeError as exc:
            print(f"Bot: Configuration error: {exc}\n")
            continue
        except requests.HTTPError as exc:
            print(f"Bot: API error: {exc}\n")
            continue
        except Exception as exc:
            print(f"Bot: Unexpected error: {exc}\n")
            continue

        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    run_chat()
