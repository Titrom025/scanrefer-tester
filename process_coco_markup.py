import cv2
import json
import os
import random

from tqdm import tqdm

from ferret_and_llava_api import describe_images, init_model
from kosmos2 import call_kosmos

DESCRIPTION_PROMPT = "Describe the object in details"

def read_scanrefer_markup(train_path, val_path):
    markup_dict = {}
    with open(train_path, 'r') as json_file:
        train_markup_list = json.load(json_file)
        for scene_markup in train_markup_list:
            scene_id = scene_markup["scene_id"]
            if scene_id not in markup_dict:
                markup_dict[scene_id] = []
            markup_dict[scene_id].append(scene_markup)
    
    with open(val_path, 'r') as json_file:
        val_markup_list = json.load(json_file)
        for scene_markup in val_markup_list:
            scene_id = scene_markup["scene_id"]
            if scene_id not in markup_dict:
                markup_dict[scene_id] = []
            markup_dict[scene_id].append(scene_markup)

    return markup_dict

def get_objects_from_coco_folder(coco_folder_path, scanrefer_markup, scene_id):
    json_file_path = os.path.join(coco_folder_path, '_annotations.coco.json')

    with open(json_file_path, 'r') as json_file:
        coco_data = json.load(json_file)

    categories = {category['id']: category['name'] for category in coco_data['categories']}

    temp_dir = "cropped"
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    scenes = {}

    for image_data in coco_data['images']:
        image_id = image_data['id']
        image_file_name = image_data['file_name']
        image_width = image_data['width']
        image_height = image_data['height']

        # print(f"Image ID: {image_id}, File Name: {image_file_name}, Width: {image_width}, Height: {image_height}")

        image_annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] == image_id]

        image_path = os.path.join(coco_folder_path, image_file_name)
        image = cv2.imread(image_path)

        for annotation in image_annotations:
            category_id = annotation['category_id']
            category_name = categories.get(category_id, 'Unknown')
            bbox = annotation['bbox']

            x, y, width, height = bbox

            cropped_object = image[int(y):int(y+height), int(x):int(x+width)]

            cropped_object_file_name = f"{category_name}_{annotation['id']}_{image_file_name}"
            cropped_object_path = os.path.join(temp_dir, cropped_object_file_name)
            cv2.imwrite(cropped_object_path, cropped_object)
            # print(f"Object: {category_name}, Bounding Box: [{x}, {y}, {width}, {height}], Saved to: {cropped_object_path}")

            description_candidates = []
            for scene_object in scanrefer_markup[scene_id]:
                if scene_object["object_name"] == category_name:
                    description_candidates.append(scene_object["description"])
            
            chosen_description = random.choice(description_candidates)
            # print(f'  - Chosen description: {chosen_description}')


            if image_id not in scenes:
                scenes[image_id] = []
            scenes[image_id].append({
                "annotation_id": annotation['id'],
                "category_id": category_id,
                "category_name": category_name,
                "image_path":  os.path.abspath(image_path),
                "crop_path": os.path.abspath(cropped_object_path),
                "description": chosen_description
            })

        # print() 
    
    return scenes

def describe_with_ferret(scenes):
    init_model(use_ferret=True)
   
    for scene_id in tqdm(scenes):
        scene_objects = scenes[scene_id]
        for obj_dict in scene_objects:
            prompt = f'You are an AI visual assistant that can analyze a single image. ' \
                     f'You receive image, describe the {obj_dict["category_name"]} in the image including color and material. ' \
                     f'There is an example of the answer format: ' \
                     f'"The trash can is black and made of plastic". Give the answer in the provided format.' \
                     f'USER: <image>\n ASSISTANT:'
            generated_descriptions = describe_images([obj_dict["crop_path"]], 
                                                      prompt, 
                                                      use_ferret=True,
                                                      replace_prompt=True)
            print(generated_descriptions)
            obj_dict["ferret_description"] = generated_descriptions[0]


def describe_with_llava(scenes):
    init_model(use_ferret=False)
   
    for scene_id in tqdm(scenes):
        scene_objects = scenes[scene_id]
        for obj_dict in scene_objects:
            prompt = f'Describe the {obj_dict["category_name"]} in the image including color and material. ' \
                     f'There is an example of the answer format: ' \
                     f'"The trash can is black and made of plastic"'
            generated_descriptions = describe_images([obj_dict["crop_path"]], 
                                                      prompt, 
                                                      use_ferret=False,
                                                      replace_prompt=True)
            print(generated_descriptions)
            obj_dict["llava_description"] = generated_descriptions[0]

def describe_with_kosmos(scenes):   
    for scene_id in tqdm(scenes):
        scene_objects = scenes[scene_id]
        for obj_dict in scene_objects:
            prompt = f'Describe the {obj_dict["category_name"]} in the image including color and material. ' \
                     f'There is an example of the answer format: ' \
                     f'"The trash can is black and made of plastic"'
            generated_description = call_kosmos(obj_dict["crop_path"], prompt)
            print(generated_description)
            obj_dict["kosmos_description"] = generated_description

def main():
    scene_id = "scene0011_00"
    # coco_markup_path = f"/media/titrom/archive/mipt/scanrefer_tester/coco_markup/{scene_id}_coco/"
    # coco_markup_path = f"/media/titrom/archive/mipt/scanrefer_tester/coco_markup/{scene_id}_coco/annotations_validated.json"
    # coco_markup_path = f"/media/titrom/archive/mipt/scanrefer_tester/coco_markup/{scene_id}_coco/annotations_with_desc_v3.json"
    # coco_markup_path = f"/media/titrom/archive/mipt/scanrefer_tester/coco_markup/{scene_id}_coco/desc_ferret_v2.json"

    coco_markup_path = f"/media/titrom/archive/mipt/scanrefer_tester/coco_markup/{scene_id}_coco/desc_ferret_llava_v2.json"
    processed_markup_path = f"/media/titrom/archive/mipt/scanrefer_tester/coco_markup/{scene_id}_coco/desc_ferret_llava_kosmos_v2.json"

    if os.path.isdir(coco_markup_path):
        train_scanref = "/media/titrom/archive/mipt/scanrefer_tester/scanrefer_markup/ScanRefer_filtered_train.json"
        val_scanref = "/media/titrom/archive/mipt/scanrefer_tester/scanrefer_markup/ScanRefer_filtered_val.json"
        scanrefer_markup = read_scanrefer_markup(train_scanref, val_scanref)
        scenes = get_objects_from_coco_folder(coco_markup_path, scanrefer_markup, scene_id)
    else:
        with open(coco_markup_path, 'r') as json_file:
            scenes = json.load(json_file)

    # describe_with_ferret(scenes)
    # describe_with_llava(scenes)
    describe_with_kosmos(scenes)
    
    with open(processed_markup_path, 'w') as json_file:
        json.dump(scenes, json_file)
    print(f'Processed markup file saved to {processed_markup_path}')


if __name__ == "__main__":
    main()