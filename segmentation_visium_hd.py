import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squidpy as sq

img = sq.im.ImageContainer(img='/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/spatial/tissue_hires_image.png')
sq.im.process(img, layer="image", method="smooth", sigma=4) 
sq.im.segment(img, layer="image_smooth", method="watershed", thresh=90, geq=False)
print(img)
# ImageContainer[shape=(3886, 6000), layers=['image', 'image_smooth', 'segmented_watershed']]

print(f"Number of segments in crop: {len(np.unique(img['segmented_watershed']))}")
# Number of segments in crop: 11


img = sq.im.ImageContainer(img='/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/spatial/tissue_hires_image.png')
sq.im.process(img, layer="image", method="smooth", sigma=4)
sq.im.segment(img, layer="image_smooth", method="watershed", geq=False)
print(f"Number of segments: {len(np.unique(img['segmented_watershed']))}")
# Number of segments: 9927

img = sq.im.ImageContainer(img='/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/spatial/tissue_hires_image.png')
sq.im.process(img, layer="image", method="smooth", sigma=4)
sq.im.segment(img, channel=0, layer="image_smooth", method="watershed", geq=False)
print(f"Number of segments: {len(np.unique(img['segmented_watershed']))}")
# Number of segments: 9927
# Number of segments: 14747 # sigma = 3
# Number of segments: 28574 # sigma = 2
# Number of segments:  # sigma = 1

fig, axes = plt.subplots(1, 2)
img.show("image", channel=0, ax=axes[0], save='/cluster/home/t116508uhn/visium_hd_hne.png')
_ = axes[0].set_title("H&E")
crop.show("segmented_watershed", cmap="jet", interpolation="none", ax=axes[1], save='/cluster/home/t116508uhn/visium_hd_segment.png')
_ = axes[1].set_title("segmentation")


############## taking too long #########################
img = sq.im.ImageContainer(img='/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/spatial/tissue_hires_image.png')
sq.im.segment(img, channel=0, layer="image", method="watershed", geq=False) # was taking long time
print(f"Number of segments in crop: {len(np.unique(img['segmented_watershed']))}")

img = sq.im.ImageContainer(img='/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/spatial/tissue_hires_image.png')
sq.im.process(img, layer="image", method="smooth")  # was taking long time
sq.im.segment(img, channel=0, layer="image_smooth", method="watershed", geq=False)
print(f"Number of segments in crop: {len(np.unique(img['segmented_watershed']))}")
# Number of segments in crop: 9927


# https://squidpy.readthedocs.io/en/stable/notebooks/examples/image/compute_segment_hne.html
# https://squidpy.readthedocs.io/en/stable/api/squidpy.im.segment.html
