## Model Setup

The fine-tuned RoBERTa model is available on Hugging Face Hub and automatically loaded by the app. See the [README](README.md) for quick setup instructions.

### Alternative: Local Model Setup

If you prefer to use a locally downloaded model instead of the Hugging Face Hub version, follow these steps:

1. **Download from Google Drive:** [Click here](https://drive.google.com/drive/folders/11ulkQ9fCmWuK8lC6InIT5U5npwjW9_I-?usp=sharing)

2. Place the following files inside a folder named `saved_model/` in the root of this project:
   - model.safetensors
   - config.json
   - tokenizer.json
   - tokenizer_config.json
   - training_args.bin

3. Update `app.py` to use the local model path instead of the Hugging Face Hub ID