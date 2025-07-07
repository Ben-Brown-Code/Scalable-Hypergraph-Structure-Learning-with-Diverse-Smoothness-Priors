import numpy as np
from skimage.segmentation import slic
import scipy.io
import matplotlib.pyplot as plt

#norm_image_dict = scipy.io.loadmat('C:/Users/btbr235/OneDrive - University of Kentucky/VSCode Projects/Hypergraphs/Hyperspectral Images PyTorch/Normalized_Image.mat')
norm_image_dict = scipy.io.loadmat('C:/Users/bentb/OneDrive - University of Kentucky/VSCode Projects/Hypergraphs/Hyperspectral Images PyTorch/Normalized_Image.mat')
norm_image = norm_image_dict['all_features_normalized_tensor'].astype(np.float32)

segments = slic(norm_image, n_segments=700, start_label=0, compactness=1)

# plt.imshow(segments, cmap='gray')
# plt.show()