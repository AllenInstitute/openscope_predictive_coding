import numpy as np
import glob
import tifffile
import os

src_dir = '/allen/aibs/technology/nicholasc/openscope/luminance_matched_images'
save_file_name = '/allen/aibs/technology/nicholasc/openscope/NATURAL_SCENES_LUMINANCE_MATCHED.npy'

file_name_list = [f for f in glob.glob(os.path.join(src_dir, '*.tiff'))]
ni = len(file_name_list)

file_name = file_name_list[0]
data = tifffile.imread(file_name)
nr, nc = data.shape

bob_index_to_data_dict = {}
for file_name in file_name_list:
    bob_index = int(os.path.basename(file_name).split('.')[0].split('_')[2])
    bob_index_to_data_dict[bob_index] = tifffile.imread(file_name)

data = np.empty((ni, nr, nc ))
for ii, img_data in bob_index_to_data_dict.items():
    data[ii, :,:] = img_data

np.save(save_file_name, data.astype(np.uint8))


