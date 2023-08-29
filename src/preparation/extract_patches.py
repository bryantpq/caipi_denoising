import numpy as np
from patchify import patchify

def extract_patches(data, dimensions, patch_size, extract_step):
    assert dimensions == len(extract_step)
    assert dimensions == len(patch_size)

    patches = patchify(data, patch_size, step=extract_step)
    patches = patches.reshape(-1, *patch_size)

    return patches
