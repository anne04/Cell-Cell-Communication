import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squidpy as sq

adata = sq.read.visium('/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/', counts_file='filtered_feature_bc_matrix.h5')
img = sq.im.ImageContainer(img='/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/spatial/tissue_hires_image.png', scale=0.07973422)
sq.im.process(img, layer="image", method="smooth", sigma=1)
sq.im.segment(img, channel=0, layer="image_smooth", method="watershed", geq=False, layer_added="segmented_watershed")
print(f"Number of segments: {len(np.unique(img['segmented_watershed']))}")
# Number of segments: 9927
# Number of segments: 14747 # sigma = 3
# Number of segments: 28574 # sigma = 2
# Number of segments: 88284 # sigma = 1


img.show("image", channel=0, save='/cluster/home/t116508uhn/visium_hd_hne.png')
img.show("segmented_watershed", cmap="jet", interpolation="none", save='/cluster/home/t116508uhn/visium_hd_segment.png')

sq.im.calculate_image_features(
    adata,
    img,
    layer="image", # or "image_smooth" ?
    features="segmentation",
    key_added="segmentation_features",
    features_kwargs={
        "segmentation": {
            "label_layer": "segmented_watershed",
            "props": ["label", "area"], # "mean_intensity"
            # "channels": [1, 2],
        }
    },
    mask_circle=True,
)
# ValueError: Expected `height` to be in interval `[0, 3886]`, found `-12`.


adata.obsm["segmentation_features"].head()

############## taking too long #########################
img = sq.im.ImageContainer(img='/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/spatial/tissue_hires_image.png')
sq.im.segment(img, channel=0, layer="image", method="watershed", geq=False) # was taking long time
print(f"Number of segments in crop: {len(np.unique(img['segmented_watershed']))}")


img = sq.im.ImageContainer(img='/cluster/projects/schwartzgroup/fatema/data/Visium_HD_Human_Colon_Cancer_square_002um_outputs/spatial/tissue_hires_image.png')
sq.im.process(img, layer="image", method="smooth")  # was taking long time
sq.im.segment(img, channel=0, layer="image_smooth", method="watershed", geq=False)
print(f"Number of segments in crop: {len(np.unique(img['segmented_watershed']))}")
# Number of segments in crop: 9927


# https://github.com/scverse/squidpy/issues/450
# https://squidpy.readthedocs.io/en/stable/notebooks/examples/image/compute_segment_hne.html
# https://squidpy.readthedocs.io/en/stable/notebooks/examples/image/compute_segmentation_features.html
# https://squidpy.readthedocs.io/en/stable/classes/squidpy.im.ImageContainer.features_segmentation.html
