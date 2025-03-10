import io
import base64
import json
import cv2
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    print(int(ytl), int(ybr), int(xtl), int(xbr))
    sub_mask = mask[int(ytl) : int(ybr), int(xtl) : int(xbr)]
    flattened = sub_mask.reshape(-1).tolist()
    flattened.extend(
        [int(xtl), int(ytl), int(xbr) - 1, int(ybr) - 1]
    )  # x-1 in case that index out of bound
    return flattened


def resize_mask(img, target_size):
    h, w = target_size
    resized_img = cv2.resize(
        img,
        (w, h),
        interpolation=cv2.INTER_NEAREST if len(img.shape) == 2 else cv2.INTER_LINEAR,
    )

    return resized_img


def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks


# Initialize your model
def init_context(context):
    context.logger.info("Init context...  0%")
    model = YOLO("model.pt")
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    model.to(device)
    context.user_data.model_handler = model
    context.logger.info("Init context...100%")


# Inference endpoint
def handler(context, event):
    context.logger.info("Run custom detic model")
    # data = json.loads(event.body) # for debug
    data = event.body
    # image_buffer = io.BytesIO(base64.b64decode(data)) # data["image"]
    image_buffer = io.BytesIO(base64.b64decode(data["image"]))
    image = cv2.imdecode(
        np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR
    )

    results = context.user_data.model_handler(image)
    result = results[0]

    boxes = result.boxes.xyxy
    masks = result.masks.data
    polygons = result.polygons
    confs = result.boxes.conf
    clss = result.cls
    names = result.names

    detections = []
    threshold = 0.45

    torch_mask = torch.stack([mask.unsqueeze(0) for mask in masks])
    torch_mask = scale_masks(torch_mask, image.shape[:2])  # N 1 H W
    for box, polygon, mask, conf, cls in zip(boxes, polygons, torch_mask, confs, clss):
        mask = mask[0].to(dtype=torch.int).cpu().numpy()
        # mask = resize_mask(mask, image.shape[:2])
        cvat_mask = to_cvat_mask(box.tolist(), mask)
        label = names[int(cls)]
        if conf >= threshold:
            # must be in this format
            detections.append(
                {
                    "confidence": str(float(conf)),
                    "label": label,
                    "points": polygon.astype(float).tolist(),
                    "mask": cvat_mask,
                    "type": "mask",
                }
            )

    print(detections)
    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
