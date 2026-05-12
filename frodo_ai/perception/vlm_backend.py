"""VLM inference backends for verify / search / stuck queries.

Usage:
    backend = make_backend("qwen")   # or "internvl"
    backend.load()                   # blocks until model is ready
    ans = backend.infer(pil_image, "Does this show a red chair?", max_tokens=5)
"""

from __future__ import annotations
from PIL import Image


class VLMBackend:
    name = "base"

    def load(self) -> None:
        raise NotImplementedError

    def infer(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError


class QwenVLBackend(VLMBackend):
    name = "Qwen2-VL-2B"
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self):
        self._model = None
        self._proc  = None

    def load(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16, device_map="auto")
        self._proc  = AutoProcessor.from_pretrained(self.MODEL_ID)

    def infer(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]}]
        text    = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        inp     = self._proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
        ids     = self._model.generate(**inp, max_new_tokens=max_tokens)
        return self._proc.batch_decode(
            [ids[0][len(inp.input_ids[0]):]],
            skip_special_tokens=True,
        )[0].strip().lower()


class InternVL2Backend(VLMBackend):
    name = "InternVL2-2B"
    MODEL_ID = "OpenGVLab/InternVL2-2B"

    def __init__(self):
        self._model = None
        self._tok   = None

    def load(self):
        from transformers import AutoModel, AutoTokenizer
        import torch
        self._model = AutoModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self._model.eval()
        self._tok = AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True)

    def infer(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        import torch
        import torchvision.transforms as T

        _MEAN = (0.485, 0.456, 0.406)
        _STD  = (0.229, 0.224, 0.225)
        tf = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])
        pixel_values = tf(image).unsqueeze(0).to(torch.float16).cuda()
        gen_cfg  = dict(max_new_tokens=max_tokens, do_sample=False)
        response = self._model.chat(
            self._tok, pixel_values, f"<image>\n{prompt}", gen_cfg
        )
        return response.strip().lower()


def make_backend(name: str) -> VLMBackend:
    n = name.lower().strip()
    if n in ("internvl", "internvl2", "internvl2-2b"):
        return InternVL2Backend()
    if n in ("qwen", "qwen2vl", "qwen2-vl", "qwen2-vl-2b"):
        return QwenVLBackend()
    raise ValueError(f"Unknown VLM backend '{name}'. Choices: qwen, internvl")
