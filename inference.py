import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class InferenceEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
        # 4-bit quantization config
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def load_model(self, model_name):
        """
        Loads a new model and tokenzier, clearing any existing model from VRAM.
        """
        self.unload_model()
        
        print(f"Loading {model_name} in 4-bit precision...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=self.quantization_config,
            trust_remote_code=True
        )
        print("Model loaded successfully.")

    def unload_model(self):
        """
        Clears the current model from VRAM.
        """
        if self.model is not None:
            print("Unloading model from VRAM...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            print("VRAM cleared.")

    def generate_response(self, messages, max_new_tokens=512):
        """
        Generates a response for the given chat messages using the loaded model.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model is currently loaded. Call load_model() first.")
            
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # deterministic generation for evaluation
                temperature=None,
                top_p=None
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
