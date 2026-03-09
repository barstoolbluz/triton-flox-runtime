import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton_python_backend_utils as pb_utils

MODEL_PATH = "/mnt/scratch/models/inferencing/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT").as_numpy().flat[0]
            input_text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

            max_tokens_tensor = pb_utils.get_input_tensor_by_name(
                request, "MAX_NEW_TOKENS"
            )
            max_new_tokens = (
                int(max_tokens_tensor.as_numpy().flat[0])
                if max_tokens_tensor is not None
                else 64
            )

            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            output_tensor = pb_utils.Tensor(
                "OUTPUT_TEXT", np.array([generated], dtype=object)
            )
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses

    def finalize(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
