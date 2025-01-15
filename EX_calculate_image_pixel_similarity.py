# import torchvision.transforms as transforms
# from PIL import Image
# import io, os
# from data.lexica_dataset import LexicaDataset

# from utils import get_image_similarity, get_pixel_mse
# from eval_PromptStealer import get_dataset


# # Get the first image from lexica dataset and save it
# # val_dataset = get_dataset("lexica")
# # val_dataset_df = val_dataset.data.to_pandas()
# # image_bytes = val_dataset_df["image"][0]['bytes']
# # image_steam = io.BytesIO(image_bytes)
# # image = Image.open(image_steam)
# # image.save(f"image.{image.format}")

# row = {
#     'ori_image_name': "image.jpg",
#     'inferred_image_save_namelist': ["00000.png","00001.png","00002.png","00003.png"]
# }

# # TODO: change like this
# '''
# for file in foler:
#     name =
#     row = {
#         'ori_image_name': f"{name}.jpg",
#         'inferred_image_save_namelist': [f"{name}_1.png","00001.png","00002.png","00003.png"]
#     }
# '''


# res_img_sim = get_image_similarity(row, "./data/test_images/ori_image", "./data/test_images/inferred_images")

# res_pixel_mse = get_pixel_mse(row, "./data/test_images/ori_image", "./data/test_images/inferred_images")

# # Print the average of each res_img_sim and res_pixel_mse
# print(f"res_img_sim: {sum(res_img_sim)/len(res_img_sim)}")
# print(f"res_pixel_mse: {sum(res_pixel_mse)/len(res_pixel_mse)}")

import os
from utils import get_image_similarity, get_pixel_mse
from eval_PromptStealer import get_dataset
directory_path = "/workspace/data1/ex2_generated_imgs/samples/"

# # List all files in the directory
# files = os.listdir(directory_path)

# # Print the list of files
# for file in files:
#     print(file)

# row = {
#     'ori_image_name': "image.jpg",
#     'inferred_image_save_namelist': ["00000.png","00001.png","00002.png","00003.png"]
# }

# res_img_sim = get_image_similarity(row, "./data/test_images/ori_image", "./data/test_images/inferred_images")
import torchvision.transforms as transforms
from PIL import Image
import io, os
val_dataset = get_dataset("lexica")
val_dataset_df = val_dataset.data.to_pandas()
image_bytes = val_dataset_df["image"][0]['bytes']
image_steam = io.BytesIO(image_bytes)
image = Image.open(image_steam)
image.save(f"./EX_genereted_img_DF/image.{image.format}")
# print(image)
# create a function that loops through all the rows in val_dataset_df and compare the generated images with the samples in the directory_path
# for each row we will compare the image with the images with the same name from the directory_path
# we will then calculate the average of the image similarity and pixel mse for each row
img_similarity_values = []
pixel_similarity_values    = []
def compare_images(row):
    try:

        image_bytes = row["image"]['bytes']
        image_steam = io.BytesIO(image_bytes)
        image = Image.open(image_steam)
        image.save(f"./EX_genereted_img_DF/image.{image.format}")
        # remove the last character from the image name
        id = row['id']
        dict = {
            'ori_image_name': f"image.{image.format}",
            'inferred_image_save_namelist': [f"{id}0.png",f"{id}1.png",f"{id}2.png",f"{id}3.png"]
        }
        res_img_sim = get_image_similarity(dict, "./EX_genereted_img_DF", directory_path)
        res_pixel_mse = get_pixel_mse(dict, "./EX_genereted_img_DF", directory_path)
        img_similarity_values.append(sum(res_img_sim)/len(res_img_sim))
        pixel_similarity_values.append(sum(res_pixel_mse)/len(res_pixel_mse))
    except:
        print("error")

val_dataset_df.apply(compare_images, axis=1)
image_similarity_average = sum(img_similarity_values)/len(img_similarity_values)
pixel_similarity_average = sum(pixel_similarity_values)/len(pixel_similarity_values)
print(f"image similarity average: {image_similarity_average}")
print(f"pixel similarity average: {pixel_similarity_average}")

    