import os
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect

MODEL_CACHE = {}

class TranslationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Enter text to translate..."}),
                "target_lang": ("STRING", {"default": "en"}),
                "source_lang": ("STRING", {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate"
    CATEGORY = "Text"

    def _load_model(self):
        """Load M2M-100 model from ComfyUI models folder or Hugging Face cache."""
        local_model_path = os.path.join("models", "translation", "facebook", "m2m100_418M")
        model_name = "facebook/m2m100_418M"

        # Use cached model if already loaded
        if model_name in MODEL_CACHE:
            return MODEL_CACHE[model_name]

        # Prefer local model
        if os.path.exists(local_model_path):
            print(f"[TranslationNode] Using local model: {local_model_path}")
            tokenizer = M2M100Tokenizer.from_pretrained(local_model_path)
            model = M2M100ForConditionalGeneration.from_pretrained(local_model_path)
        else:
            print("[TranslationNode] Local model not found, downloading from Hugging Face...")
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        MODEL_CACHE[model_name] = (tokenizer, model)
        return tokenizer, model

    def translate(self, text, target_lang, source_lang):
        """Perform translation using M2M-100 model."""
        if not text.strip():
            return ("⚠️ Empty input text.",)

        # Auto-detect source language
        if source_lang == "auto":
            try:
                source_lang = detect(text)
            except Exception:
                source_lang = "en"

        tokenizer, model = self._load_model()
        tokenizer.src_lang = source_lang
        encoded = tokenizer(text, return_tensors="pt")
        generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return (translated,)

NODE_CLASS_MAPPINGS = {"TranslationNode": TranslationNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TranslationNode": "🈯 Translation Node (Offline)"}
