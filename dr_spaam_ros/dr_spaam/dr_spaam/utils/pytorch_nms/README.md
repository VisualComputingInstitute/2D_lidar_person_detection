# Torchvision support for NMS

Note: Since the publication of this repository, NMS support has been included as part of torchvision. Therefore you might want to use this implementation instead:
https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

This repository might still be of interest if you need the index in the `keep` list of the highest-scoring box overlapping each input box.

# CUDA implementation of NMS for PyTorch.


This repository has a CUDA implementation of NMS for PyTorch 1.4.0.

The code is released under the BSD license however it also includes parts of the original implementation from [Fast R-CNN](https://github.com/rbgirshick/py-faster-rcnn) which falls under the MIT license (see LICENSE file for details).

The code is experimental and has not be thoroughly tested yet; use at your own risk. Any issues and pull requests are welcome.

## Installation

```
python setup.py install
```

## Usage

Example:
```
from nms import nms

keep, num_to_keep, parent_object_index = nms(boxes, scores, overlap=.5, top_k=200)
```

The `nms` function takes a (N,4) tensor of `boxes` and associated (N) tensor of `scores`, sorts the bounding boxes by score and selects boxes using Non-Maximum Suppression according to the given `overlap`. It returns the indices of the `top_k` with the highest score. Bounding boxes are represented using the standard (left,top,right,bottom) coordinates representation.

`keep` is the list of indices of kept bounding boxes. Note that the tensor size is always (N) however only the first `num_to_keep` entries are valid.

For each input box, the (N) tensor `parent_object_index` contains the index (1-based) in the `keep` list of the highest-scoring box overlapping this box. This can be useful to group input boxes that are related to the same object. The index 0 represents a background box which has been ignored due to `top_k`.

Currently there is a hard-limit of 64,000 input boxes. You can change the constant `MAX_COL_BLOCKS` in `nms_kernel.cu` to increase this limit.

