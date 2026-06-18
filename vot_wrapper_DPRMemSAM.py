import sys

import numpy as np
from PIL import Image

from custom_drmsam_tracker import CustomDAM4SAMTracker
import torch

import utils.vot_helper as vot

import random
import os
import yaml

# Everything is read from this config (model size, RAM / DRM / long-term memory).
_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "DRPMemSAM_config.yaml")) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)

def get_vot_mask(masks_list, image_width, image_height):
    id_ = 1
    masks_multi = np.zeros((image_height, image_width), dtype=np.float32)
    for mask in masks_list:
        m = make_full_size(mask, (image_width, image_height))
        masks_multi[m>0] = id_
        id_ += 1
    return masks_multi


@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():
    run = config.get("run") or {}
    tracker_name = f"sam{run['sam']}pp-{run['size']}"   # e.g. "sam21pp-T"
    run_args = run.get("args") or []
    ram_cfg = config.get("ram") or {}
    tracker = CustomDAM4SAMTracker(
        tracker_name,
        memory_stride=ram_cfg.get("stride", 1),
        num_maskmem=ram_cfg.get("num_frames"),
        drm_min_stride=1,
        apply_postprocessing="--no_postprocessing" not in run_args,
        drm_config=config.get("drm"),
        longterm_config=config.get("longterm"),
        extra_memory_config=config.get("extra_memory"),
    )

    # Sequential mode: VOT runs this wrapper once per object independently
    # !TODO: Need to adapt this with the simultaneous to get faster resutls
    handle = vot.VOT("mask", multiobject=False)
    region = handle.region()

    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = Image.open(imagefile)
    init_mask = make_full_size(region, (image.width, image.height))
    tracker.initialize(image, init_mask)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        image = Image.open(imagefile)
        outputs = tracker.track(image)
        handle.report(outputs['pred_mask'])

if __name__ == "__main__":
    main()
