import os

import folder_paths

from PIL import Image
import numpy as np
import torch

from ultralytics import YOLO

folder_paths.folder_names_and_paths["yolov8"] = ([os.path.join(folder_paths.models_dir, "yolov8")], folder_paths.supported_pt_extensions)

class YOLOv8SegmentationNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "model_name": (folder_paths.get_filename_list("yolov8"), ),
                "class_ind": ("INT", {"default": 0}),
                "combine": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                }),
                "instance_ind": ("INT", {"default": 0}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE", )
    RETURN_NAMES = ("segmentation", "annotation", )
    FUNCTION = "segment"
    CATEGORY = "aichemy"
    
    def tensor2PIL(self, tensor: torch.Tensor) -> Image:
        """
        Convert a PyTorch tensor to a PIL Image.

        Args:
            tensor (torch.Tensor): The input tensor to be converted.

        Returns:
            PIL.Image.Image: The converted PIL Image.

        Tensor should be in the range of [0, 1].
        """
        return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))
    
    def PIL2tensor(self, image: Image) -> torch.Tensor:
        """
        Convert a PIL Image to a PyTorch tensor.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            torch.Tensor: The converted PyTorch tensor.

        Resulting tensor is in the range of [0, 1].
        """
        return torch.tensor(np.array(image).astype(np.float32) / 255.0)
    
    def crop_padding(
            self, mask: np.ndarray, 
            orig_shape: tuple[int, int], 
            tol: float = 0.1,
        ) -> np.ndarray:
        """
        Remove the padding from the mask produced by YOLOv8.
        Replicates parts of the original function from 
        ultralytics.utils.ops.scale_image that behaved 
        weirdly when imported. The tolerance was implemented
        to match the original function.

        Args:
            mask (np.ndarray): The mask to be cropped.
            orig_shape (tuple): The original shape of the image, HxW.
            tol (float, optional): The tolerance value. Defaults to 0.1.

        Returns:
            np.ndarray: The cropped and padded mask.
        """
        mask_shape = mask.shape
        # orig factor is the un-padded one
        factor = min(
            mask_shape[0] / orig_shape[0], 
            mask_shape[1] / orig_shape[1]
        )  
        pad = (
            (mask_shape[0] - orig_shape[0] * factor) / 2, # H
            (mask_shape[1] - orig_shape[1] * factor) / 2, # W
        )
        mask = mask[
            int(round(pad[0] - tol)) : int(round(mask_shape[0] - pad[0] + tol)), # H crop
            int(round(pad[1] - tol)) : int(round(orig_shape[1] - pad[1] + tol)), # W crop
        ]
        return mask
    
    def segment(
            self, 
            image: torch.Tensor, # batch_size x H x W x 3
            model_name: str, 
            class_ind: int, 
            combine: bool, 
            instance_ind: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Segments an image using a specified YOLOv8 model and selects 
        the specified class or class instance segmentation mask

        Args:
            image (np.ndarray): The input image batch of shape BxHxWxC.
            model_name (str): The name of the model to be used.
            class_ind (int): The index of the selected segmentation class.
            combine (bool): A flag indicating whether to combine all instances of the specified class.
            instance_ind (int): The index of the selected instance of the specified class (if combine is False).

        Returns:
            tuple: A tuple containing two tensors:
                - The segmented mask as a torch.Tensor of shape BxHxWxC.
                - The plotted annotation result as a torch.Tensor of shape BxHxWxC.
        """
        model = YOLO(f'{os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')  # load a custom model
        result = model(self.tensor2PIL(image[0]))[0] #only one image at a time
        
        class_mask_inds = torch.where(result.boxes.data[:, 5] == class_ind)
        class_masks = result.masks.data[class_mask_inds]

        class_mask = torch.any(class_masks, dim=0) if combine or instance_ind >= class_masks.shape[0] else class_masks[instance_ind]

        cropped_mask = self.crop_padding(class_mask, result.orig_shape)
        cropped_mask = self.tensor2PIL(cropped_mask)

        full_mask = cropped_mask.resize((result.orig_shape[1], result.orig_shape[0]))

        return (
            torch.unsqueeze(self.PIL2tensor(full_mask), 0),
            torch.unsqueeze(self.PIL2tensor(result.plot()[:, :, ::-1]), 0), #BGR to RGB
        )


NODE_CLASS_MAPPINGS = {
    "aichemyYOLOv8Segmentation": YOLOv8SegmentationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "aichemyYOLOv8Segmentation": 'YOLOv8 Segmentaion',
}