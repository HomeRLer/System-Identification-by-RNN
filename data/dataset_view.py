import os
import shutil

import pandas as pd

dataset_dir = "datasets_6000"
# output_dir = "dasets_6000"
file_list = os.listdir(dataset_dir)
# os.mkdir(output_dir)
for file in file_list:
    file_dir = dataset_dir + "/" + file
    df = pd.read_csv(file_dir)
    length = df.shape[0]
    print(file + ": " + str(length))

    # if length == 6000:
    #     shutil.copyfile(file_dir, output_dir + "/" + file)

