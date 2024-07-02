
try:
    import torch
    from transformers import (
        BlipProcessor,
        BlipForConditionalGeneration,
        Blip2Processor,
        Blip2ForConditionalGeneration,
    )
    from PIL import Image
    from typing import Union
    from pathlib import Path
    from .base import BaseImageBackend

    class BaseBlipImageBackend(BaseImageBackend):
        default_blip = None

        processor_class = BlipProcessor
        model_class = BlipForConditionalGeneration
        init_model = True

        def __init__(self, options: dict):
            self.options = options
            if not self.default_blip and not "model_path" in options:
                raise ValueError("No default model path or model path provided")
            
            dev_override = options.get("device", None)
            self.device = torch.device(dev_override or f"cuda" if torch.cuda.is_available() else "cpu")
            self.model_path = options.get("model_path", self.default_blip)
            self.conditional_captioning = options.get("conditional_captioning", False)
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            if self.init_model:
                self.intialize()

        def intialize(self):
            self.processor = self.processor_class.from_pretrained(self.model_path)
            self.model = self.model_class\
                .from_pretrained(self.model_path, torch_dtype=self.dtype, **self.options.get("model_options", {}))\
                .to(self.device)
            self.model.eval()

        def caption(self, fp: Union[Image.Image, str, Path, bytes], formats: Union[list, tuple]=None, caption_override: str=None):
            """
                Opens and captions the given image file.
                Works with django's File, or ImageFile.
                You can pass either a path, the bytes of the image, or the image itself.
            """
            if not isinstance(fp, Image.Image):
                pil = Image.open(fp, formats=formats).convert('RGB')
            else:
                pil = fp.convert('RGB')

            override = caption_override or self.conditional_captioning
            if override:
                # Image with conditional captioning, IE 'A photography of'
                if callable(override):
                    override = override(pil)
                image = self.processor(pil, override, return_tensors="pt")
            else:
                image = self.processor(pil, return_tensors="pt")

            image.to(self.model.device, self.dtype)

            caption = self.model.generate(**image)
            return self.processor.batch_decode(caption, skip_special_tokens=True)
        
    class BlipBaseImageBackend(BaseBlipImageBackend):
        default_blip = "Salesforce/blip-image-captioning-base"

    class BlipLargeImageBackend(BaseBlipImageBackend):
        default_blip = "Salesforce/blip-image-captioning-large"

    class Blip2Opt3BImageBackend(BaseBlipImageBackend):
        default_blip = "Salesforce/blip2-opt-2.7b"
        processor_class = Blip2Processor
        model_class = Blip2ForConditionalGeneration

    class Blip2Opt7BImageBackend(Blip2Opt3BImageBackend):
        default_blip = "Salesforce/blip2-opt-6.7b"
        processor_class = Blip2Processor
        model_class = Blip2ForConditionalGeneration

    class Blip2Opt7BCocoImageBackend(Blip2Opt3BImageBackend):
        default_blip = "Salesforce/blip2-opt-6.7b-coco"
        processor_class = Blip2Processor
        model_class = Blip2ForConditionalGeneration

    class Blip2FlanT5XLImageBackend(Blip2Opt3BImageBackend):
        default_blip = "Salesforce/blip2-flan-t5-xl"
        processor_class = Blip2Processor
        model_class = Blip2ForConditionalGeneration

    class Blip2FlanT5XXLImageBackend(Blip2Opt3BImageBackend):
        default_blip = "Salesforce/blip2-flan-t5-xxl"
        processor_class = Blip2Processor
        model_class = Blip2ForConditionalGeneration

except ImportError:
    BaseBlipImageBackend = None
    BlipBaseImageBackend = None
    BlipLargeImageBackend = None
    Blip2Opt3BImageBackend = None
    Blip2Opt7BImageBackend = None
    Blip2Opt7BCocoImageBackend = None
    Blip2FlanT5XLImageBackend = None
    Blip2FlanT5XXLImageBackend = None
