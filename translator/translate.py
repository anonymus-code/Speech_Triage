from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import(
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from IndicTransToolkit.IndicTransToolkit.processor import IndicProcessor

class CustomLLM():
    def __init__(self):
        self.model_name = "ai4bharat/indictrans2-indic-en-1B"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.DEVICE)
        self.ip = IndicProcessor(inference=True)
        self.src_lang = "hin_Deva"
        self.tgt_lang = "eng_latn"


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        batch = self.ip.preprocess_batch(
            [prompt],
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
        )
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.DEVICE)

    # Generate translations using the model
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with self.tokenizer.as_target_tokenizer():
            generated_tokens = self.tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations = self.ip.postprocess_batch(generated_tokens, lang=self.tgt_lang)
        
        return translations[0]
    
