import os
import numpy as np

origin_path = "./ori_data"
target_path = "./tar_data"

domains = ["20240425"]

np.random.seed(42)
for domain in domains:
    ori_dir_path = os.path.join(origin_path, domain)
    categories = os.listdir(ori_dir_path)
    categories = [category.replace(".npy", "") for category in categories]
    for category in categories:
        ori_numpy_data_path = os.path.join(ori_dir_path, f"{category}.npy")
        ori_numpy_data = np.load(ori_numpy_data_path)
        train_length = int(len(ori_numpy_data) * 0.8)
        np.random.shuffle(ori_numpy_data)
        numpy_data_split_dic = {
            "train": ori_numpy_data[:train_length],
            "val": ori_numpy_data[train_length:],
        }
        for dtype in ["train", "val"]:
            numpy_data_split = numpy_data_split_dic[dtype]
            print(domain, category, dtype, len(numpy_data_split))
            tar_numpy_dir_path = os.path.join(target_path, domain, dtype, category)
            os.makedirs(tar_numpy_dir_path, exist_ok=True)
            for idx, sample in enumerate(numpy_data_split):
                tar_numpy_data_path = os.path.join(tar_numpy_dir_path, f"{domain}_{category}_{dtype}_{idx}.npy")
                np.save(tar_numpy_data_path, sample)
