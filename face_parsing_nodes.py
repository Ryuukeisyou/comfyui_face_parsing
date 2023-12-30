import os
import cv2
import numpy as np
import folder_paths
import torch
from torch import Tensor
import torch.nn as nn
import matplotlib.cm as cm
import torchvision.transforms as T
from torchvision.transforms import functional
from ultralytics import YOLO

models_path = folder_paths.models_dir
face_parsing_path = os.path.join(models_path, "face_parsing")
cur_dir = os.path.dirname(__file__)

class FaceBBoxDetectorLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("ultralytics_bbox")
        face_detect_models = list(filter(lambda x: 'face' in x, files))
        bboxs = ["bbox/" + x for x in face_detect_models]
        return {
            "required": {
                "model_name": (bboxs, {})
            }
        }
    
    RETURN_TYPES = ("BBOX_DETECTOR",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, model_name):
        model_path = folder_paths.get_full_path("ultralytics", model_name)
        model = YOLO(model_path) # type: ignore
        return (model, ) 
    
class FaceBBoxDetect:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_detector": ("BBOX_DETECTOR", {}),
                "image": ("IMAGE", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1,
                    "step": 0.01
                }),
                "dilation": ("INT", {
                    "default": 8,
                    "min": -512,
                    "max": 512,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("BBOX_LIST",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox_detector: YOLO, image: Tensor, threshold: float, dilation: int):
        results = []
        transform = T.ToPILImage()
        for item in image:
            image_pil = transform(item.permute(2, 0, 1))
            pred = bbox_detector(image_pil, conf=threshold)
            bboxes = pred[0].boxes.xyxy.cpu()
            for bbox in bboxes:
                bbox[0] = bbox[0] - dilation
                bbox[1] = bbox[1] - dilation
                bbox[2] = bbox[2] + dilation
                bbox[3] = bbox[3] + dilation
            results.append(bboxes)
        return (results,)

class BBoxListItemSelect:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_list": ("BBOX_LIST", {}),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("BBOX",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox_list: list, index: int):
        item = bbox_list[index if index < len(bbox_list) - 1 else len(bbox_list) - 1]
        return (item,)

class BBoxResize:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
                "width_old": ("INT", {}),
                "height_old": ("INT", {}),
                "width": ("INT", {}),
                "height": ("INT", {}),
            },
        }

    RETURN_TYPES = ("BBOX",)
    # RETURN_NAMES = ("any")

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "face_parsing"

    def main(self, bbox: Tensor, width_old: int, height_old: int, width: int, height: int):
        newBbox = bbox.clone()
        bbox_values = newBbox[0].float()
        l = bbox_values[0] / width_old * width
        t = bbox_values[1] / height_old * height
        r = bbox_values[2] / width_old * width
        b = bbox_values[3] / height_old * height

        newBbox[0][0] = l
        newBbox[0][1] = t
        newBbox[0][2] = r
        newBbox[0][3] = b
        return (newBbox,)

class ImageCropWithBBox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox: Tensor, image: Tensor):
        image_permuted = image.permute(0, 3, 1, 2)
        bbox_item = bbox[0]
        bbox_int = bbox_item.int()
        l = bbox_int[0]
        t = bbox_int[1]
        r = bbox_int[2]
        b = bbox_int[3]
        cropped_image = functional.crop(image_permuted, t, l, b-t, r-l) # type: ignore
        final = cropped_image.permute(0, 2, 3, 1)
        return (final,)

class ImagePadWithBBox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
                "width": ("INT", {}),
                "height": ("INT", {}),
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox: Tensor, width: int, height: int, image: Tensor):
        image_permuted = image.permute(0, 3, 1, 2)
        bbox_item = bbox[0]
        bbox_int = bbox_item.int()
        l = bbox_int[0]
        t = bbox_int[1]
        r = bbox_int[2]
        b = bbox_int[3]
        cropped_image = functional.pad(image_permuted, [l, t, width - r, height - b]) # type: ignore
        final = cropped_image.permute(0, 2, 3, 1)
        return (final,)

class ImageInsertWithBBox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
                "image_src": ("IMAGE", {}),
                "image": ("IMAGE", {}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox: Tensor, image_src: Tensor, image: Tensor):
        bbox_item = bbox[0]
        bbox_int = bbox_item.int()
        l = bbox_int[0]
        t = bbox_int[1]
        r = bbox_int[2]
        b = bbox_int[3]
        
        image_permuted = image.permute(0, 3, 1, 2)
        resized = functional.resize(image_permuted, [b - t, r - l]) # type: ignore

        _, h, w, c = image_src.shape
        padded = functional.pad(resized, [l, t, w - r, h - b])  # type: ignore

        src_permuted = image_src.permute(0, 3, 1, 2)
        result = torch.where(padded == 0, src_permuted, padded)

        final = result.permute(0, 2, 3, 1)
        return (final,)

class ImageListSelect:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1
                })
            }
        }
 
    RETURN_TYPES = ("IMAGE", )

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, image: Tensor, index: int):
        return (image[index].unsqueeze(0),)

class ImageSize:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height")

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "face_parsing"

    def main(self, image: Tensor):
        w = image.shape[2]
        h = image.shape[1]
        return (w, h)

class ImageResizeCalculator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "target_size": ("INT", {
                    "default": 512,
                    "min": 1,
                    "step": 1
                }),
                "force_8x": ("BOOLEAN", {
                    "default": True,
                })
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT",)
    RETURN_NAMES = ("width", "height", "min",)

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "face_parsing"

    def main(self, image: Tensor, target_size: int, force_8x: bool):
        w = image.shape[2]
        h = image.shape[1]
        ratio = h * 1.0 / w
        if (w >= h):
            w_new = target_size
            h_new = target_size * ratio
            if force_8x:
                w_new = int(w_new / 8) * 8
                h_new = int(h_new / 8) * 8
            # scale = target_size * 1.0 / w
            # scale_back = w / target_size * 1.0
            return (w_new, int(h_new), h_new, )
        else:
            w_new = target_size / ratio
            h_new = target_size
            if force_8x:
                w_new = int(w_new / 8) * 8
                h_new = int(h_new / 8) * 8
            # scale = target_size * 1.0 / h
            # scale_back = h / target_size * 1.0
            return (int(w_new), h_new, w_new, )

class FaceParsingModelLoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }
 
    RETURN_TYPES = ("FACE_PARSING_MODEL", )

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self):
        from transformers import AutoModelForSemanticSegmentation
        model = AutoModelForSemanticSegmentation.from_pretrained(face_parsing_path)
        return (model,)

class FaceParsingProcessorLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }
    
    RETURN_TYPES = ("FACE_PARSING_PROCESSOR", )

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self):
        from transformers import SegformerImageProcessor
        processor = SegformerImageProcessor.from_pretrained(face_parsing_path)
        return (processor,)

class FaceParse:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FACE_PARSING_MODEL", {}),
                "processor": ("FACE_PARSING_PROCESSOR", {}),
                "image": ("IMAGE", {})
            }
        }

    RETURN_TYPES = ("IMAGE", "FACE_PARSING_RESULT",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, model, processor, image: Tensor):
        images = []
        results = []
        transform = T.ToPILImage()
        colormap = cm.get_cmap('viridis', 19)

        for item in image:
            size = item.shape[:2]
            inputs = processor(images=transform(item.permute(2, 0, 1)), return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=size,
                mode="bilinear",
                align_corners=False)
            
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            pred_seg_np = pred_seg.detach().numpy().astype(np.uint8)
            results.append(torch.tensor(pred_seg_np))
            
            colored = colormap(pred_seg_np)
            colored_sliced = colored[:,:,:3] # type: ignore
            images.append(torch.tensor(colored_sliced))

        return (torch.cat(images, dim=0).unsqueeze(0), torch.cat(results, dim=0).unsqueeze(0),)
    
class FaceParsingResultsParser:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("FACE_PARSING_RESULT", {}),
                "background": ("BOOLEAN", {"default": False}),
                "skin": ("BOOLEAN", {"default": True}),
                "nose": ("BOOLEAN", {"default": True}),
                "eye_g": ("BOOLEAN", {"default": True}),
                "r_eye": ("BOOLEAN", {"default": True}),
                "l_eye": ("BOOLEAN", {"default": True}),
                "r_brow": ("BOOLEAN", {"default": True}),
                "l_brow": ("BOOLEAN", {"default": True}),
                "r_ear": ("BOOLEAN", {"default": True}),
                "l_ear": ("BOOLEAN", {"default": True}),
                "mouth": ("BOOLEAN", {"default": True}),
                "u_lip": ("BOOLEAN", {"default": True}),
                "l_lip": ("BOOLEAN", {"default": True}),
                "hair": ("BOOLEAN", {"default": True}),
                "hat": ("BOOLEAN", {"default": True}),
                "ear_r": ("BOOLEAN", {"default": True}),
                "neck_l": ("BOOLEAN", {"default": True}),
                "neck": ("BOOLEAN", {"default": True}),
                "cloth": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(
            self, 
            result: Tensor,
            background: bool,
            skin: bool,
            nose: bool,
            eye_g: bool,
            r_eye: bool,
            l_eye: bool,
            r_brow: bool,
            l_brow: bool,
            r_ear: bool,
            l_ear: bool,
            mouth: bool,
            u_lip: bool,
            l_lip: bool,
            hair: bool,
            hat: bool,
            ear_r: bool,
            neck_l: bool,
            neck: bool,
            cloth: bool):
        masks = []
        for item in result:
            mask = torch.zeros(item.shape, dtype=torch.uint8)
            if (background):
                mask = mask | torch.where(item == 0, 1, 0)
            if (skin):
                mask = mask | torch.where(item == 1, 1, 0)
            if (nose):
                mask = mask | torch.where(item == 2, 1, 0)    
            if (eye_g):
                mask = mask | torch.where(item == 3, 1, 0)  
            if (r_eye):
                mask = mask | torch.where(item == 4, 1, 0) 
            if (l_eye):
                mask = mask | torch.where(item == 5, 1, 0) 
            if (r_brow):
                mask = mask | torch.where(item == 6, 1, 0) 
            if (l_brow):
                mask = mask | torch.where(item == 7, 1, 0) 
            if (r_ear):
                mask = mask | torch.where(item == 8, 1, 0) 
            if (l_ear):
                mask = mask | torch.where(item == 9, 1, 0) 
            if (mouth):
                mask = mask | torch.where(item == 10, 1, 0) 
            if (u_lip):
                mask = mask | torch.where(item == 11, 1, 0) 
            if (l_lip):
                mask = mask | torch.where(item == 12, 1, 0)   
            if (hair):
                mask = mask | torch.where(item == 13, 1, 0) 
            if (hat):
                mask = mask | torch.where(item == 14, 1, 0) 
            if (ear_r):
                mask = mask | torch.where(item == 15, 1, 0) 
            if (neck_l):
                mask = mask | torch.where(item == 16, 1, 0) 
            if (neck):
                mask = mask | torch.where(item == 17, 1, 0)   
            if (cloth):
                mask = mask | torch.where(item == 18, 1, 0)         
            masks.append(mask.float())
        final = torch.cat(masks, dim=0).unsqueeze(0)
        return (final,)

class MaskComposite:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("MASK", {}),
                "source": ("MASK", {}),
                "operation": (["multiply", "add", "subtract", "and", "or", "xor"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    # RETURN_NAMES = ("any")

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "face_parsing"

    def main(self, destination: Tensor, source: Tensor, operation: str):
        mask_result = destination
        if (operation == 'multiply'):
            mask_result = mask_result * source
        if (operation == 'add'):
            mask_result = mask_result + source
        if (operation == 'subtract'):
            mask_result = mask_result - source
        if (operation == 'and'):
            mask_result = mask_result & source
        if (operation == 'or'):
            mask_result = mask_result | source
        if (operation == 'xor'):
            mask_result = mask_result ^ source
        return (mask_result,)

class MaskListComposite:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {}),
                "operation": (["multiply", "add", "and", "or", "xor"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    # RETURN_NAMES = ("any")

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "face_parsing"

    def main(self, mask: Tensor, operation: str):
        mask_result = mask[0]
        for item in mask[1:]:
            if (operation == 'multiply'):
                mask_result = mask_result * item
            if (operation == 'add'):
                mask_result = mask_result + item
            if (operation == 'and'):
                mask_result = mask_result & item
            if (operation == 'or'):
                mask_result = mask_result | item
            if (operation == 'xor'):
                mask_result = mask_result ^ item
        return (mask_result.unsqueeze(0),)

class MaskListSelect:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {}),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1
                })
            }
        }
 
    RETURN_TYPES = ("MASK", )

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, mask: Tensor, index: int):
        return (mask[index].unsqueeze(0),)

class GuidedFilter:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {
                    "default": 3,
                    "min": 0,
                    "step": 1
                }),
                "eps": ("FLOAT", {
                    "default": 125,
                    "min": 0,
                    "step": 1,
                }),
            },
            "optional": {
                "guide": ("IMAGE", {
                    "default": None
                })
            }
        }
 
    RETURN_TYPES = ("IMAGE", )
    # RETURN_NAMES = ("image")

    FUNCTION = "guided_filter"

    #OUTPUT_NODE = False

    CATEGORY = "face_parsing"

    def guided_filter(self,
            image: Tensor,
            radius: int,
            eps: float,
            guide: Tensor | None = None):
        results = []
        for item in image:
            image_cv2 = cv2.cvtColor(item.mul(255).byte().numpy(), cv2.COLOR_RGB2BGR)
            guide_cv2 = image_cv2 if guide is None else  cv2.cvtColor(guide.numpy(), cv2.COLOR_RGB2BGR)
            result_cv2 = cv2.ximgproc.guidedFilter(guide_cv2, image_cv2, radius, eps)
            result_cv2_rgb = cv2.cvtColor(result_cv2, cv2.COLOR_BGR2RGB)
            result = torch.tensor(result_cv2_rgb).float().div(255)
            results.append(result)
        return (torch.cat(results, dim=0).unsqueeze(0),)

class ColorAdjust:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
                "brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": -255,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
                "hue": ("FLOAT", {
                    "default": 0,
                    "min": -0.5,
                    "max": 0.5,
                    "step": 0.001,
                    "round": 0.001,
                    "display": "number"
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "face_parsing"

    def main(self, 
             image: Tensor, 
             contrast: float = 1, 
             brightness: float = 1, 
             saturation: float = 1,
             hue: float = 0,
             gamma: float = 1):

        permutedImage = image.permute(0, 3, 1, 2)

        if (contrast != 1):
            permutedImage = functional.adjust_contrast(permutedImage, contrast)

        if (brightness != 1):
            permutedImage = functional.adjust_brightness(permutedImage, brightness)

        if (saturation != 1):
            permutedImage = functional.adjust_saturation(permutedImage, saturation)

        if (hue != 0):
            permutedImage = functional.adjust_hue(permutedImage, hue)

        if (gamma != 1):
            permutedImage = functional.adjust_gamma(permutedImage, gamma)

        result = permutedImage.permute(0, 2, 3, 1)

        return (result,)


NODE_CLASS_MAPPINGS = {
    'FaceBBoxDetectorLoader(FaceParsing)': FaceBBoxDetectorLoader,
    'FaceBBoxDetect(FaceParsing)': FaceBBoxDetect,
    'BBoxListItemSelect(FaceParsing)': BBoxListItemSelect,
    'BBoxResize(FaceParsing)': BBoxResize,
    'ImageSize(FaceParsing)': ImageSize,
    'ImageResizeCalculator(FaceParsing)': ImageResizeCalculator,
    'ImageCropWithBBox(FaceParsing)': ImageCropWithBBox,
    'ImagePadWithBBox(FaceParsing)':ImagePadWithBBox,
    'ImageInsertWithBBox(FaceParsing)':ImageInsertWithBBox,
    'ImageListSelect(FaceParsing)':ImageListSelect,
    'FaceParsingProcessorLoader(FaceParsing)': FaceParsingProcessorLoader,
    'FaceParsingModelLoader(FaceParsing)': FaceParsingModelLoader,
    'FaceParse(FaceParsing)': FaceParse,
    'FaceParsingResultsParser(FaceParsing)': FaceParsingResultsParser,
    'MaskListComposite(FaceParsing)':MaskListComposite,
    'MaskListSelect(FaceParsing)':MaskListSelect,
    'MaskComposite(FaceParsing)':MaskComposite,
    'GuidedFilter(FaceParsing)': GuidedFilter,
    'ColorAdjust(FaceParsing)': ColorAdjust,  
}