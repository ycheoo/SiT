import os
import numpy as np

max_size = 3 * 224 * 224

def concat_npy(save_path, ori_name, extracted_npy_data_arrays):
    count, pre_index = 0, 0
    npy_data_concat = np.array([])
    for index, npy_data in enumerate(extracted_npy_data_arrays):
        if len(npy_data) + len(npy_data_concat) > max_size and len(npy_data_concat) != 0:
            npy_file_name = f"{ori_name}_concat_{pre_index}-{index}_{count}"
            npy_path_name = os.path.join(save_path, npy_file_name)
            np.save(npy_data_concat, npy_path_name)
            count, pre_index = count + 1, index
            npy_data_concat = np.array([])
        npy_data_concat = np.concatenate((npy_data_concat, npy_data))
    if len(npy_data_concat) != 0:
        npy_file_name = f"{ori_name}_concat_{pre_index}-{len(extracted_npy_data_arrays) - 1}_{count}"
        npy_path_name = os.path.join(save_path, npy_file_name)
        np.save(npy_data_concat, npy_path_name)