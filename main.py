import os
from dataset_test import dataset_test

# Use correct relative paths based on where your script is
base_dir = os.path.join(os.getcwd(), 'DRIVE', 'Test')
file_path_im = os.path.join(base_dir, 'images')         # e.g., C:/.../DRIVE/Test/images/
file_path_manual = os.path.join(base_dir, '1st_manual') # e.g., C:/.../DRIVE/Test/1st_manual/

im_postfix = '.tif'
ma_postfix = '.gif'

manual_path = "C:/Users/Manjitha.K/Downloads/retinal_vessel_segmentation/DRIVE/Test/1st_manual/01_manual1.gif"

print("Looking for:", manual_path)
print("Files in folder:")
print(os.listdir(file_path_manual))

dataset_test(file_path_im, file_path_manual, im_postfix, ma_postfix)
