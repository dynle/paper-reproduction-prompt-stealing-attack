import os
from utils import get_image_similarity, get_pixel_mse
from eval_PromptStealer import get_dataset
import torchvision.transforms as transforms
from PIL import Image
import io, os

directory_path = "directory_path"

val_dataset = get_dataset("lexica")
val_dataset_df = val_dataset.data.to_pandas()
image_bytes = val_dataset_df["image"][0]['bytes']
image_steam = io.BytesIO(image_bytes)
image = Image.open(image_steam)
image.save(f"./EX_genereted_img_DF/image.{image.format}")

img_similarity_values = []
pixel_similarity_values = []
def compare_images(row):
    try:
        # Convert image bytes to image
        image_bytes = row["image"]['bytes']
        image_steam = io.BytesIO(image_bytes)
        image = Image.open(image_steam)
        image.save(f"./EX_genereted_img_DF/image.{image.format}")
        id = row['id']

        # Input format
        dict = {
            'ori_image_name': f"image.{image.format}",
            'inferred_image_save_namelist': [f"{id}0.png",f"{id}1.png",f"{id}2.png",f"{id}3.png"]
        }

        # Get image similarity and pixel mse
        res_img_sim = get_image_similarity(dict, "./EX_genereted_img_DF", directory_path)
        res_pixel_mse = get_pixel_mse(dict, "./EX_genereted_img_DF", directory_path)
        img_similarity_values.append(sum(res_img_sim)/len(res_img_sim))
        pixel_similarity_values.append(sum(res_pixel_mse)/len(res_pixel_mse))
    except:
        print("error")

# Calculate image similarity and pixel mse
val_dataset_df.apply(compare_images, axis=1)
image_similarity_average = sum(img_similarity_values)/len(img_similarity_values)
pixel_similarity_average = sum(pixel_similarity_values)/len(pixel_similarity_values)
print(f"image similarity average: {image_similarity_average}")
print(f"pixel similarity average: {pixel_similarity_average}")