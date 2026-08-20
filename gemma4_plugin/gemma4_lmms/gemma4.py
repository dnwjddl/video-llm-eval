"""Gemma 4 (E2B/E4B) wrapper for lmms-eval.

Adapted from lmms_eval/models/simple/gemma3.py, with the model loader switched
to Gemma 4's AutoModelForMultimodalLM (falls back to AutoModelForImageTextToText
on older transformers). Videos are passed as file paths and sampled to frames by
the Gemma 4 processor.
"""

import os
import re
import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForMultimodalLM as _AutoMultimodal
except ImportError:  # transformers < Gemma 4 support
    from transformers import AutoModelForImageTextToText as _AutoMultimodal

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.models.model_utils.media_encoder import encode_image_to_data_url

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1605632
DEFAULT_MAX_FRAMES = 32


class Gemma4(lmms):
    """
    Gemma 4 (E2B/E4B) Model
    https://huggingface.co/google/gemma-4-E4B-it
    """

    def __init__(
        self,
        pretrained: str = "google/gemma-4-E4B-it",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_num_frames: int = DEFAULT_MAX_FRAMES,
        interleave_visuals: Optional[bool] = False,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = _AutoMultimodal.from_pretrained(pretrained, **model_kwargs).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)

        self._config = self._model.config
        self._max_length = kwargs.get("max_length", 2048)
        if hasattr(self._model, "tie_weights"):
            self._model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        self.model.eval()

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Not implemented for Gemma4.")

    def flatten(self, input: List[List]) -> List:
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _encode_image_data_url(self, image: Image.Image) -> str:
        return encode_image_to_data_url(
            image,
            image_format="JPEG",
            mime_type="image/jpeg",
            convert_rgb=True,
            quality=85,
        )

    @staticmethod
    def _video_total_frames(path: str) -> Optional[int]:
        try:
            from decord import VideoReader

            return len(VideoReader(path))
        except Exception:
            pass
        try:
            import cv2

            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return n if n > 0 else None
        except Exception:
            return None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            batched_messages = []
            # Short clips can have fewer frames than max_num_frames; asking the
            # processor for more than the video holds raises ValueError.
            effective_num_frames = self.max_num_frames
            has_video = False
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                for visual in visual_list[i]:
                    try:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                            if not os.path.exists(visual):
                                eval_logger.warning(f"Video file not found: {visual}")
                                continue
                            has_video = True
                            total = self._video_total_frames(visual)
                            if total is not None and total < effective_num_frames:
                                effective_num_frames = total
                            processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                        elif isinstance(visual, Image.Image):
                            processed_visuals.append({"type": "image", "image": self._encode_image_data_url(visual), "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                    except Exception as e:
                        eval_logger.error(f"Failed to process visual: {e}")
                        continue

                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
                batched_messages.append(message)

            template_kwargs = {"num_frames": effective_num_frames} if has_video else {}
            # Decoders disagree on frame counts for variable-frame-rate clips, so a
            # pre-count can still overshoot what the processor's own loader sees.
            # On "num_frames exceeds total_num_frames=N", retry with the N the
            # processor reported.
            while True:
                try:
                    inputs = self.processor.apply_chat_template(
                        batched_messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True,
                        **template_kwargs,
                    )
                    break
                except ValueError as e:
                    m = re.search(r"total_num_frames=(\d+)", str(e))
                    reported = int(m.group(1)) if m else 0
                    current = template_kwargs.get("num_frames")
                    if has_video and current is not None and 0 < reported < current:
                        eval_logger.warning(f"num_frames={current} exceeds actual {reported}; retrying with {reported}")
                        template_kwargs["num_frames"] = reported
                        continue
                    raise
            inputs = inputs.to(self.model.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
