# Task 5: Mental Health Support Chatbot

## Quick Start

### 1. Install Dependencies
```bash
pip install transformers[torch] datasets accelerate
pip install streamlit  # Optional, for web interface
```

### 2. Fine-tune the Model (Optional)
Training takes 5-15 minutes on GPU, 1-2 hours on CPU:
```bash
python task5_mental_health_chatbot.py --train
```

This downloads the EmpatheticDialogues dataset (~350MB) and fine-tunes DistilGPT2.

### 3. Chat with the Chatbot

**CLI Mode (default):**
```bash
python task5_mental_health_chatbot.py --chat
```

**Streamlit Web Mode:**
```bash
streamlit run task5_mental_health_chatbot.py
```

## Features

✅ **Empathetic Responses** - Fine-tuned on real human empathetic dialogues  
✅ **Crisis Detection** - Identifies harmful keywords and provides resources  
✅ **Multiple Interfaces** - CLI or Streamlit web app  
✅ **No API Keys** - Runs locally on your machine  
✅ **Mental Health Resources** - Emergency numbers and crisis hotlines  

## Crisis Resources

**If someone is in immediate danger:**

**US:**
- National Suicide Prevention Lifeline: **988**
- Crisis Text Line: Text **HOME** to **741741**

**International:**
- [IASP Crisis Centers](https://www.iasp.info/resources/Crisis_Centres/)

## Dataset

- **EmpatheticDialogues** by Facebook AI Research
- 25K real human conversations with empathetic responses
- Automatically downloaded and cached on first run

## Technical Details

| Component | Details |
|-----------|---------|
| Base Model | DistilGPT2 (82M parameters) |
| Framework | Hugging Face Transformers |
| Method | Causal Language Model Fine-tuning |
| Batch Size | 8 |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Max Length | 128 tokens |

## Performance

- **Training Time:** 5-15 min (GPU), 1-2 hours (CPU)
- **Model Size:** ~240MB
- **Response Time:** <1 second
- **Memory:** ~2GB VRAM (GPU)

## Limitations

⚠️ This chatbot provides **emotional support only**, not clinical therapy  
⚠️ Should NOT be used as a substitute for mental health treatment  
⚠️ Always recommends professional help for serious concerns  

## Troubleshooting

**Q: Out of memory error during training**  
A: Reduce BATCH_SIZE in the script (default: 8) or use CPU mode

**Q: Model not found for chat**  
A: Train the model first: `python task5_mental_health_chatbot.py --train`

**Q: Slow responses**  
A: Responses are slower on CPU. GPU is highly recommended.

**Q: Internet connection required?**  
A: Only for initial dataset download. Chat mode works offline.
