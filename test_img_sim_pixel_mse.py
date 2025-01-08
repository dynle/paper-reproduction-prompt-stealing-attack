import torchvision.transforms as transforms
from PIL import Image
import io, os
from data.lexica_dataset import LexicaDataset

from utils import get_image_similarity, get_pixel_mse
from eval_PromptStealer import get_dataset

# Get the first image from lexica dataset and save it
# val_dataset = get_dataset("lexica")
# val_dataset_df = val_dataset.data.to_pandas()
# image_bytes = val_dataset_df["image"][0]['bytes']
# image_steam = io.BytesIO(image_bytes)
# image = Image.open(image_steam)
# image.save(f"image.{image.format}")

row = {
    'ori_image_name': "image.jpg",
    'inferred_image_save_namelist': ["00000.png","00001.png","00002.png","00003.png"]
}

# TODO: change like this
'''
for file in foler:
    name =
    row = {
        'ori_image_name': f"{name}.jpg",
        'inferred_image_save_namelist': [f"{name}_1.png","00001.png","00002.png","00003.png"]
    }
'''


res_img_sim = get_image_similarity(row, "./data/test_images/ori_image", "./data/test_images/inferred_images")

res_pixel_mse = get_pixel_mse(row, "./data/test_images/ori_image", "./data/test_images/inferred_images")

# Print the average of each res_img_sim and res_pixel_mse
print(f"res_img_sim: {sum(res_img_sim)/len(res_img_sim)}")
print(f"res_pixel_mse: {sum(res_pixel_mse)/len(res_pixel_mse)}")