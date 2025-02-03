import os
import cv2
import matplotlib.colors
import numpy as np
import folder_paths
import torch
from torch import Tensor
import torch.nn as nn
import matplotlib
import torchvision.transforms as T
from torchvision import ops
from torchvision.transforms import functional

models_path = folder_paths.models_dir
face_parsing_path = os.path.join(models_path, "face_parsing")
cur_dir = os.path.dirname(__file__)

class BBoxDetectorLoader:
    def __init__(self):
        pass

    @classmethod
    def GetModelList(cls) -> list[str]:
        files = folder_paths.get_filename_list("ultralytics_bbox")
        # face_detect_models = list(filter(lambda x: 'face' in x, files))
        # bboxs = ["bbox/" + x for x in face_detect_models]
        bboxs = ["bbox/" + x for x in files]
        return bboxs
    
    
    @classmethod
    def INPUT_TYPES(cls):
        bboxs = cls.GetModelList()
        return {
            "required": {
                "model_name": (bboxs, {})
            }
        }
    
    RETURN_TYPES = ("BBOX_DETECTOR",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, model_name):
        from ultralytics import YOLO
        model_path = folder_paths.get_full_path("ultralytics_bbox", model_name.split("/")[-1])
        model = YOLO(model_path) # type: ignore
        return (model, ) 

class BBoxDetect:
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
                }),
                "dilation_ratio": ("FLOAT", {
                    "default": 0.2,
                    "min": 0,
                    "max": 1,
                    "step": 0.01
                }),
                "by_ratio": ("BOOLEAN", {
                    "default": False,
                })
            }
        }

    RETURN_TYPES = ("BBOX_LIST", "INT", )

    RETURN_NAMES = ("BBOX_LIST", "count")

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox_detector, image: Tensor, threshold: float, dilation: int, dilation_ratio: float, by_ratio: bool):
        results = []
        transform = T.ToPILImage()
        for item in image:
            image_pil = transform(item.permute(2, 0, 1))
            pred = bbox_detector(image_pil, conf=threshold)
            bboxes = pred[0].boxes.xyxy.cpu()
            for bbox in bboxes:
                l, t, r, b = bbox
                final_dilation = dilation if not by_ratio else int(max(int(b-t), int(r-l)) * dilation_ratio)
                bbox[0] = bbox[0] - final_dilation
                bbox[1] = bbox[1] - final_dilation
                bbox[2] = bbox[2] + final_dilation
                bbox[3] = bbox[3] + final_dilation
                bbox[0] = bbox[0] if bbox[0] > 0 else 0
                bbox[1] = bbox[1] if bbox[1] > 0 else 0
                bbox[2] = bbox[2] if bbox[2] < item.shape[1] else item.shape[1]
                bbox[3] = bbox[3] if bbox[3] < item.shape[0] else item.shape[0]
                results.append(bbox)
        return (results, len(results))

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
        bbox_values = newBbox.float()
        l = bbox_values[0] / width_old * width
        t = bbox_values[1] / height_old * height
        r = bbox_values[2] / width_old * width
        b = bbox_values[3] / height_old * height

        newBbox[0] = l
        newBbox[1] = t
        newBbox[2] = r
        newBbox[3] = b
        return (newBbox,)

class BBoxDecompose:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("l", "t", "r", "b")

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "face_parsing"

    def main(self, bbox: Tensor):
        bbox_int = bbox.round().int()
        l = int(bbox_int[0])
        t = int(bbox_int[1])
        r = int(bbox_int[2])
        b = int(bbox_int[3])
        return (l, t, r, b)

class LatentCropWithBBox:
    def __init__(self):
        pass
     
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
                "samples": ("LATENT", {}),
            }
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox: Tensor, samples: dict):
        bbox_int = bbox.round().int() // 8
        l = int(bbox_int[0])
        t = int(bbox_int[1])
        r = int(bbox_int[2])
        b = int(bbox_int[3])

        samples_value = samples['samples']
        cropped_samples_value = functional.crop(samples_value, t, l, (b - t), (r - l))
        result = {'samples': cropped_samples_value}

        # if ('noise_mask' in samples and keep_mask):
        #     noise_mask_value = samples['noise_mask']
        #     cropped_noise_mask_value = functional.crop(noise_mask_value, t, l, b - t, r -l)
        #     result['noise_mask'] = cropped_noise_mask_value

        return (result,)

class LatentInsertWithBBox:
    def __init__(self):
        pass
     
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
                "samples_src": ("LATENT", {}),
                "samples": ("LATENT", {}),
            }
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox: Tensor, samples_src: dict, samples: dict):
        bbox_int = bbox.round().int() // 8

        l = int(bbox_int[0])
        t = int(bbox_int[1])
        r = int(bbox_int[2])
        b = int(bbox_int[3])

        samples_src_val : Tensor = samples_src['samples']
        w = samples_src_val.shape[3]
        h = samples_src_val.shape[2]

        mask = torch.zeros(samples_src_val.shape)
        mask[:, :, t:b, l:r] = 1
        
        samples_val : Tensor = samples['samples']
        resized_samples_val = functional.resize(samples_val, [(b - t),  (r - l)])
        padded_samples_val = functional.pad(resized_samples_val, [l, t, (w - r), (h - b)])

        final_samples_val = torch.where(mask == 0, samples_src_val, padded_samples_val)
        result = {'samples': final_samples_val}

        return (result,)

class LatentSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, samples: dict):
        shape = samples['samples'].shape
        h = shape[2] * 8
        w = shape[3] * 8
        return (w, h,)

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
        results = []
        image_permuted = image.permute(0, 3, 1, 2)
        for image_item in image_permuted:
            bbox_int = bbox.round().int()
            l = bbox_int[0]
            t = bbox_int[1]
            r = bbox_int[2]
            b = bbox_int[3]
            cropped_image = functional.crop(image_item, t, l, b-t, r-l) # type: ignore
            result = cropped_image.permute(1, 2, 0)
            results.append(result)
        try: 
            results = torch.stack(results, dim=0)
        except:
            pass
        return (results,)

class ImageCropWithBBoxList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_list": ("BBOX_LIST", {}),
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox_list: list[Tensor], image: Tensor):
        results = []
        image_permuted = image.permute(0, 3, 1, 2)
        for image_item in image_permuted:
            for bbox in bbox_list:
                bbox_int = bbox.round().int()
                l = bbox_int[0]
                t = bbox_int[1]
                r = bbox_int[2]
                b = bbox_int[3]
                cropped_image = functional.crop(image_item, t, l, b-t, r-l) # type: ignore
                result = cropped_image.permute(1, 2, 0)
                results.append(result)
        try: 
            results = torch.stack(results, dim=0)
        except:
            pass
        return (results,)

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
        bbox_int = bbox.round().int()
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
        bbox_int = bbox.round().int()
        l = bbox_int[0]
        t = bbox_int[1]
        r = bbox_int[2]
        b = bbox_int[3]
        
        image_permuted = image.permute(0, 3, 1, 2)
        resized = functional.resize(image_permuted, [b - t, r - l]) # type: ignore

        _, h, w, c = image_src.shape
        padded = functional.pad(resized, [l, t, w - r, h - b])  # type: ignore

        src_permuted = image_src.permute(0, 3, 1, 2)
        mask = torch.zeros(src_permuted.shape)
        mask[:, :, t:b, l:r] = 1
        result = torch.where(mask == 0, src_permuted, padded)

        final = result.permute(0, 2, 3, 1)
        return (final,)

class ImageResizeWithBBox:
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
        bbox_int = bbox.round().int()
        l = int(bbox_int[0])
        t = int(bbox_int[1])
        r = int(bbox_int[2])
        b = int(bbox_int[3])

        resized_image = functional.resize(image.permute(0, 3, 1, 2), [b - t, r - l]).permute(0, 2, 3, 1)
        return (resized_image,)

class ImageListSelect:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1
                })
            }
        }
 
    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE", )

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, images, index):
        index = index[0]
        if images is Tensor:
            return (images[index].unsqueeze(0),)
        else:
            return (images[index],)

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
                }),
                "force_64x": ("BOOLEAN", {
                    "default": False,
                })
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width", "height", "width_old", "height_old", "min", "scale", "scale_r")

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "face_parsing"

    def main(self, image: Tensor, target_size: int, force_8x: bool, force_64x: bool):
        w = image[0].shape[1]
        h = image[0].shape[0]

        ratio = h * 1.0 / w
        if (w >= h):
            w_new = target_size
            h_new = target_size * ratio
            if force_64x:
                w_new = int(w_new / 64) * 64
                h_new = int(h_new / 64) * 64
            elif force_8x:
                w_new = int(w_new / 8) * 8
                h_new = int(h_new / 8) * 8
            # scale = target_size * 1.0 / w
            # scale_back = w / target_size * 1.0
            return (w_new, int(h_new), w, h, h_new, w_new * 1.0 / w, w * 1.0 / w_new)
        else:
            w_new = target_size / ratio
            h_new = target_size
            if force_64x:
                w_new = int(w_new / 64) * 64
                h_new = int(h_new / 64) * 64
            elif force_8x:
                w_new = int(w_new / 8) * 8
                h_new = int(h_new / 8) * 8
            # scale = target_size * 1.0 / h
            # scale_back = h / target_size * 1.0
            return (int(w_new), h_new, w, h, w_new, h_new * 1.0 / h, h * 1.0 / h_new )

class FaceParsingModelLoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cpu", "cuda"], {
                    "default": "cpu"
                    })
            }
        }
 
    RETURN_TYPES = ("FACE_PARSING_MODEL", )

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, device: str):
        from transformers import AutoModelForSemanticSegmentation
        model = AutoModelForSemanticSegmentation.from_pretrained(face_parsing_path)
        if device == "cuda" and torch.cuda.is_available():
            model.cuda()
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
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FACE_PARSING_RESULT",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, model, processor, image: Tensor):
        images = []
        results = []
        transform = T.ToPILImage()
        colormap = matplotlib.colormaps['viridis']
        device = model.device

        for item in image:
            size = item.shape[:2]
            inputs = processor(images=transform(item.permute(2, 0, 1)), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=size,
                mode="bilinear",
                align_corners=False)
            
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            pred_seg_np = pred_seg.cpu().detach().numpy().astype(np.uint8)
            results.append(torch.tensor(pred_seg_np))
            
            norm = matplotlib.colors.Normalize(0, 18)
            pred_seg_np_normed = norm(pred_seg_np)
            colored = colormap(pred_seg_np_normed)
            colored_sliced = colored[:,:,:3] # type: ignore
            images.append(torch.tensor(colored_sliced))

        images_out = torch.stack(images, dim=0)
        results_out = torch.stack(results, dim=0)
        return (images_out, results_out,)

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
        masks_out = torch.stack(masks, dim=0)
        return (masks_out,)

# class SkinDetectTraditional:
#     def __init__(self):
#         pass

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "mode": (["RGB", "YCrCb", "HSV"], {})
#             }
#         }

#     RETURN_TYPES = ("MASK", )

#     FUNCTION = "main"

#     CATEGORY = "face_parsing"

#     def main(self, image: Tensor, mode: str):
#         image = image.permute(0, 3, 1, 2).squeeze(0).mul(255).byte()
#         if (mode == 'YCrCb'):
#             mask = skin_detection_utils.detect_skin_YcrCb(image, 0)
#         elif (mode == 'HSV'):
#             mask = skin_detection_utils.detect_skin_HSV(image, 0)
#         else:
#             mask = skin_detection_utils.detect_skin_RGB(image, 0)[0]
#         return (mask,)

class MaskBorderDissolve:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {}),
                "size": ("INT", {
                    "default": 10
                    }),
                "kernel_size": ("INT", {
                    "default": 5,
                    }),
                "sigma": ("FLOAT", {
                    "default": 0,
                    }),
            }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, mask: Tensor, size: int, kernel_size: int, sigma: float):
        results = []
        if kernel_size % 2 == 0:
            kernel_size += 1
        for mask_item in mask:
            white = torch.ones_like(mask_item, dtype=torch.float32)
            if len(white.shape) == 2:
                white = white.unsqueeze(0)
            _, h, w = white.shape
            white[:, size : h - 1 - size, size : w - 1 - size] = 0
            blurred = functional.gaussian_blur(white, kernel_size=[kernel_size, kernel_size], sigma=None if sigma == 0 else [sigma, sigma])
            result = mask_item - blurred
            if len(mask_item.shape) == 2 and len(result.shape) == 3:
                result = result.squeeze(0)
            result = torch.clamp(result, min=0)
                        
            results.append(result)
        try: 
            results = torch.stack(results, dim=0)
        except:
            pass
        return (results,)
    
class MaskBlackOut:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        directions = ["left", "top", "right", "bottom"]
        return {
            "required": {
                "mask": ("MASK", {}),
                "direction": (directions, {
                    "default": "top"
                    }),
                "position": ("INT", {
                    "default": 10,
                    }),
            }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, mask: Tensor, direction: str, position: int):
        results = []
        for mask_item in mask:
            copy = mask_item.clone()
            h, w = mask_item.shape
            match direction:
                case "left":
                    copy[:, :position] = 0
                case "top":
                    copy[:position, :] = 0
                case "right":
                    copy[:, position:] = 0
                case "bottom":
                    copy[position:, :] = 0
            result = torch.clamp(copy, min=0)
                        
            results.append(result)
        try: 
            results = torch.stack(results, dim=0)
        except:
            pass
        return (results,)

class MaskCropWithBBox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
                "mask": ("MASK", {}),
            }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox: Tensor, mask: Tensor):
        results = []
        mask_permuted = mask.permute(0, 3, 1, 2)
        for mask_item in mask_permuted:
            bbox_int = bbox.round().int()
            l = bbox_int[0]
            t = bbox_int[1]
            r = bbox_int[2]
            b = bbox_int[3]
            cropped_mask = functional.crop(mask_item, t, l, b-t, r-l) # type: ignore
            result = cropped_mask.permute(1, 2, 0)
            results.append(result)
        try: 
            results = torch.stack(results, dim=0)
        except:
            pass
        return (results,)

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
        mask_result = torch.clamp(mask_result, min=0)
        return (mask_result,)

class MaskBatchComposite:
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

class MaskToBBoxList:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "pad": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1
                }),
            },
        }
 
    RETURN_TYPES = ("BBOX_LIST", )
    # RETURN_NAMES = ("BBOX_LIST")

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "face_parsing"

    def main(self, mask: Tensor, pad: int):
        try:
            result = ops.masks_to_boxes(mask)
        except:
            result = None
        if pad != 0 and result is not None:
            for item in result:
                item[0] = item[0] - pad
                item[1] = item[1] - pad
                item[2] = item[2] + pad
                item[3] = item[3] + pad

        return ([item for item in result] if result is not None else None,)

class MaskInsertWithBBox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
                "image_src": ("IMAGE", {}),
                "mask": ("MASK", {}),
            }
        }
    
    RETURN_TYPES = ("MASK",)

    FUNCTION = "main"

    CATEGORY = "face_parsing"

    def main(self, bbox: Tensor, image_src: Tensor, mask: Tensor):
        bbox_int = bbox.round().int()
        l = bbox_int[0]
        t = bbox_int[1]
        r = bbox_int[2]
        b = bbox_int[3]

        resized = functional.resize(mask, [b - t, r - l]) # type: ignore

        _, h, w, c = image_src.shape
        padded = functional.pad(resized, [l, t, w - r, h - b])  # type: ignore

        return (padded,)

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
        return (torch.stack(results, dim=0),)

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
    'BBoxDetectorLoader(FaceParsing)': BBoxDetectorLoader,
    'BBoxDetect(FaceParsing)': BBoxDetect,
    'BBoxListItemSelect(FaceParsing)': BBoxListItemSelect,
    'BBoxResize(FaceParsing)': BBoxResize,
    'BBoxDecompose(FaceParsing)': BBoxDecompose,

    # 'LatentCropWithBBox(FaceParsing)': LatentCropWithBBox,
    # 'LatentInsertWithBBox(FaceParsing)': LatentInsertWithBBox,
    # 'LatentSize(FaceParsing)': LatentSize,

    'ImageCropWithBBox(FaceParsing)': ImageCropWithBBox,
    'ImageCropWithBBoxList(FaceParsing)': ImageCropWithBBoxList,
    'ImagePadWithBBox(FaceParsing)':ImagePadWithBBox,
    'ImageInsertWithBBox(FaceParsing)':ImageInsertWithBBox,
    'ImageResizeWithBBox(FaceParsing)':ImageResizeWithBBox,
    'ImageListSelect(FaceParsing)':ImageListSelect,
    'ImageSize(FaceParsing)': ImageSize,
    'ImageResizeCalculator(FaceParsing)': ImageResizeCalculator,
    
    'FaceParsingProcessorLoader(FaceParsing)': FaceParsingProcessorLoader,
    'FaceParsingModelLoader(FaceParsing)': FaceParsingModelLoader,
    'FaceParse(FaceParsing)': FaceParse,
    'FaceParsingResultsParser(FaceParsing)': FaceParsingResultsParser,

    # 'SkinDetectTraditional(FaceParsing)':SkinDetectTraditional,
    
    'MaskBorderDissolve(FaceParsing)':MaskBorderDissolve,
    'MaskBlackOut(FaceParsing)': MaskBlackOut,
    'MaskCropWithBBox(FaceParsing)': MaskCropWithBBox,
    'MaskBatchComposite(FaceParsing)': MaskBatchComposite,
    'MaskListSelect(FaceParsing)': MaskListSelect,
    'MaskComposite(FaceParsing)': MaskComposite,
    'MaskToBBoxList(FaceParsing)': MaskToBBoxList,
    'MaskInsertWithBBox(FaceParsing)': MaskInsertWithBBox,
    
    'GuidedFilter(FaceParsing)': GuidedFilter,
    'ColorAdjust(FaceParsing)': ColorAdjust,  
}
