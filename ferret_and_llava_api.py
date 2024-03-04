import argparse
import base64
import json
import os
import re
import requests
import time
import torch
import shutil

import cv2

from io import BytesIO
from tqdm import tqdm

from PIL import Image

DEFAULT_REGION_FEA_TOKEN = "<region_fea>"
COORDS_TAG = "<target_coords>"

IMAGE_W = 1000
IMAGE_H = 1000
LOG_FILE = None


def reset_log():
    with open(LOG_FILE, 'w'):
        pass


def print_and_log(message):
    with open(LOG_FILE, 'a') as f_clip:
        f_clip.write(message + '\n')
    print(message)


def prepare_out_path(filename, obj_id=None):
    path_parts = filename.split(os.path.sep)
    new_first_folder = "frames_with_bboxes"
    path_parts[0] = new_first_folder
    if obj_id is not None:
        path_parts[-1] = path_parts[-1].replace('.jpg', f'_{obj_id}.jpg')
    return os.path.join(*path_parts)


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def get_frames_from_video(videopath):
    paths = []
    count = 1
    video_name = os.path.splitext(os.path.basename(videopath))[0]
    TMP_DIR = os.path.join('frames_from_video/', video_name)
    if os.path.exists(TMP_DIR):
        for path in os.listdir(TMP_DIR):
            if '.jpg' in path:
                paths.append(os.path.join(TMP_DIR, f"frame_{count}.jpg"))
                count += 1
        print(f'Frames from video found in cache: {len(paths)}')
        return paths

    print('Getting frames from video')   
    os.makedirs(TMP_DIR)
    
    vidcap = cv2.VideoCapture(videopath)
    success,image = vidcap.read()
    while success:
        frame_path = os.path.join(TMP_DIR, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, image)
        paths.append(frame_path)
        success,image = vidcap.read()
        count += 1
    return paths


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)

    return code


def prepare_image(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()

    return img_b64_str, image.size


def get_coords_from_text(text, image_size):
    pattern = r'\[(\d+,\s*\d+,\s*\d+,\s*\d+)\]'
    matches = re.findall(pattern, text)

    bboxes = [tuple(map(int, match.split(','))) for match in matches]

    for i, bbox in enumerate(bboxes):
        bboxes[i] = convert_coords(bbox, image_size, to_model_format=False)
    
    return bboxes


def get_position_from_text(output_message, position_map):
    for pos in position_map.keys():
        if pos in output_message:
            return position_map[pos]
    return None

def draw_frame_id(image2draw, frame_id):
    cv2.putText(image2draw, f'Frame: {frame_id}', 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (255, 0, 255), 5, cv2.LINE_AA
    )


def draw_boxes(image2draw, bbox_tuples):
    for bbox in bbox_tuples:
        cv2.rectangle(image2draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)


def draw_position(image2draw, coords, color, position_label=None):    
    text_x = coords[0]
    text_y = coords[1] - 15
    if text_x > 0.7 * image2draw.shape[1]:
        text_x = round(0.7 * image2draw.shape[1])

    cv2.rectangle(image2draw, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
    
    if position_label is not None:
        cv2.putText(image2draw, position_label, (text_x, text_y), 2, 1, color)


def draw_relation(image2draw, coords1, coords2, relation, color):    
    text_x = (coords1[0] + coords2[0]) // 2
    text_y = (coords1[1] + coords2[1]) // 2
    
    if text_x > 0.7 * image2draw.shape[1]:
        text_x = round(0.7 * image2draw.shape[1])

    # cv2.rectangle(image2draw, (coords1[0], coords1[1]), (coords1[2], coords1[3]), color, 2)
    # cv2.rectangle(image2draw, (coords2[0], coords2[1]), (coords2[2], coords2[3]), color, 2)
    cv2.line(
        image2draw,
        ((coords1[0] + coords1[2]) // 2, (coords1[1] + coords1[3]) // 2),
        ((coords2[0] + coords2[2]) // 2, (coords2[1] + coords2[3]) // 2),
        color, 2
    )
    
    cv2.putText(image2draw, relation, (text_x, text_y), 4, 1, color)


def find_indices_in_order(str_list, STR):
    indices = []
    i = 0
    while i < len(STR):
        for element in str_list:
            if STR[i:i+len(element)] == element:
                indices.append(str_list.index(element))
                i += len(element) - 1
                break
        i += 1
    return indices


def generate_mask_for_feature(coor, raw_w, raw_h, mask=None):
    if mask is not None:
        assert mask.shape[0] == raw_w and mask.shape[1] == raw_h
    coor_mask = torch.zeros((raw_w, raw_h))
    # Assume it samples a point.
    if len(coor) == 2:
        # Define window size
        span = 10
        # Make sure the window does not exceed array bounds
        x_min = max(0, coor[0] - span)
        x_max = min(raw_w, coor[0] + span + 1)
        y_min = max(0, coor[1] - span)
        y_max = min(raw_h, coor[1] + span + 1)
        coor_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
        assert (coor_mask==1).any(), f"coor: {coor}, raw_w: {raw_w}, raw_h: {raw_h}"
    elif len(coor) == 4:
        # Box input or Sketch input.
        coor_mask = torch.zeros((raw_w, raw_h))
        coor_mask[coor[0]:coor[2]+1, coor[1]:coor[3]+1] = 1
        if mask is not None:
            coor_mask = coor_mask * mask
    # coor_mask = torch.from_numpy(coor_mask)
    # pdb.set_trace()
    assert len(coor_mask.nonzero()) != 0
    return coor_mask.tolist()

def format_region_prompt(prompt, coords):
    if coords is not None:
        prompt = prompt.replace(COORDS_TAG, f'{coords} {DEFAULT_REGION_FEA_TOKEN}')
    return prompt

def convert_coords(coords, image_size, to_model_format=True):
    x1, y1, x2, y2 = coords
    if to_model_format:
        x1 = max(0, round(x1 * IMAGE_W / image_size[0]))
        y1 = max(0, round(y1 * IMAGE_H / image_size[1]))
        x2 = min(999, round(x2 * IMAGE_W / image_size[0]))
        y2 = min(999, round(y2 * IMAGE_H / image_size[1]))
    else:
        x1 = max(0, round(x1 / IMAGE_W * image_size[0]))
        y1 = max(0, round(y1 / IMAGE_H * image_size[1]))
        x2 = min(image_size[0]-1, round(x2 / IMAGE_W * image_size[0]))
        y2 = min(image_size[1]-1, round(y2 / IMAGE_H * image_size[1]))
    
    coords = [x1, y1, x2, y2]
    return coords
    
def send_promt_with_image(image_path, prompt_text, masks=[], log_status=False, replace_prompt=False):
    prepared_image, image_size = prepare_image(image_path)
    
    # prompt = 'A chat between a human and an AI that understands visuals. ' \
    #         'In images, [x, y] denotes points: top-left [0, 0], bottom-right [width-1, height-1]. ' \
    #         'Increasing x moves right; y moves down. Bounding box: [x1, y1, x2, y2]. ' \
    #         f'Image size: {IMAGE_W}x{IMAGE_H}. Follow instructions.  USER: <image>\n ' \
    #         f'{prompt_text} ASSISTANT:'    

    if not replace_prompt:
        prompt = f'You are an AI visual assistant that can analyze a single image. ' \
                 f'You receive image and specific object locations within the image are given, along with detailed coordinates. ' \
                 f'In images, [x, y] denotes points: top-left [0, 0], bottom-right [width-1, height-1]. ' \
                 f'Increasing x moves right; y moves down. Bounding box: [x1, y1, x2, y2]. Image size: {IMAGE_W}x{IMAGE_H}. ' \
                 f'Follow instructions.  USER: <image>\n ' \
                 f'{prompt_text} ASSISTANT:'    
    else:
        prompt = prompt_text

    headers = {'User-Agent': 'FERRET Client'}
    pload = {
        'model': model_name, 
        'prompt': prompt, 
        'temperature': 0.2, 
        'top_p': 0.7, 
        'max_new_tokens': 512, 
        'stop': '</s>',
        'images': [prepared_image],
        'region_masks': masks
    }

    response = requests.post(worker_addr + "/worker_generate_stream",
        headers=headers, json=pload, stream=True, timeout=10)

    output_message = ''

    if log_status:
        # print_and_log(f'Input: {prompt_text}')
        print_and_log('Generating answer...')
    t_start = time.time()
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            if data["error_code"] == 0:
                output = data["text"][len(prompt):].strip()
                output = post_process_code(output)
                output_message = output + "▌"
            else:
                output_message = data["text"] + f" (error_code: {data['error_code']})"
                print(f'Error code: {output_message}')
                break

    t_end = time.time()
    if log_status:
        print_and_log(f'Final answer: {output_message}')
        print_and_log(f'Time to generate: {round(t_end - t_start, 1)}s')

    return output_message


def detect_objects(image_paths, markup_data, prompt_text):
    print('Detecting object locations')
    total_positions = correct_pos = incorrect_pos = 0
    incorrect_detections = []
    for image_idx, image_path in enumerate(image_paths, start=1):
        image_idx_str = str(image_idx)
        if image_idx_str in markup_data:
            for obj_id, markup_obj in enumerate(markup_data[image_idx_str], start=1):
                print_and_log(f'Handling {image_path} obj {obj_id+1}')
                coords = [
                    markup_obj["x1"], markup_obj["y1"],
                    markup_obj["x2"], markup_obj["y2"]
                ]
                print_and_log(f'{coords}')
                target_position = markup_obj["target"]

                image_draw_path = prepare_out_path(image_path, obj_id)
                image2draw = cv2.imread(image_path)
                draw_frame_id(image2draw, image_idx)
                cv2.imwrite(image_draw_path, image2draw)

                coords_model = convert_coords(coords, image_size, to_model_format=True)
                prompt_text = format_region_prompt(prompt_text, coords_model)

                
                masks = [generate_mask_for_feature(coords, raw_w=IMAGE_W, raw_h=IMAGE_H)]

                output_message = send_promt_with_image(
                    image_path, prompt_text, masks
                )
                with Image.open(image_path) as image:
                    image_size = image.size

                bboxes = get_coords_from_text(output_message, image_size)

                position_map = {
                    'floor': 'on the floor', 
                    'table': 'on the table', 
                    'desk': 'on the table',
                    'dresser': 'on the table',
                    'counter': 'on the table',
                    'chair': 'on the chair'
                }
                position = get_position_from_text(output_message, position_map)

                image_draw_path = prepare_out_path(image_path, obj_id)
                
                if os.path.exists(os.path.dirname(image_draw_path)):
                    shutil.rmtree(os.path.dirname(image_draw_path))
                os.makedirs(os.path.dirname(image_draw_path))

                image2draw = cv2.imread(image_draw_path)
                draw_boxes(image2draw, bboxes)

                correct_position = False
                if target_position is not None:
                    correct_position = target_position == position
                    if target_position != position:
                        cv2.putText(image2draw, f'Target: {target_position}', (coords[0], coords[1] - 40), 2, 1, color)
                        position_label = f"Predicted: {position}"
                        color = (0, 0, 205)
                    else:
                        position_label = f"Correct: {position}"
                        color = (0, 205, 0)
                else:
                    position_label = "Unknown position"
                    color = (128, 0, 128)
                    
                draw_position(image2draw, coords, color, position_label)
                cv2.imwrite(image_draw_path, image2draw)

                print_and_log(f"Position correct: {correct_position}")
                if correct_position:
                    correct_pos += 1
                else:
                    incorrect_pos += 1
                    incorrect_detections.append(f'{image_path}, obj: {obj_id}')
                
                total_positions += 1
        else:
            image_draw_path = prepare_out_path(image_path)
            if not os.path.exists(os.path.dirname(image_draw_path)):
                os.makedirs(os.path.dirname(image_draw_path), exist_ok=True)
            image2draw = cv2.imread(image_path)
            draw_frame_id(image2draw, image_idx)
            cv2.imwrite(image_draw_path, image2draw)
            print_and_log(f'No coords on {image_path}')

        print_and_log('')
    
    if len(incorrect_detections):
        print_and_log(f'Incorrect positions:')
    for det in incorrect_detections:
        print_and_log(det)
            
    print_and_log(f'Total object positions: {total_positions}')
    print_and_log(f'Correct object positions: {correct_pos}')
    print_and_log(f'Incorrect object positions: {incorrect_pos}')
    print_and_log(f'Accuracy: {round(correct_pos / total_positions * 100, 1)}')


def detect_relations(image_paths, markup_data):
    print('Detecting object relations')
    ignore_objects = ["drawer", "person", "handle"]

    image_draw_path = prepare_out_path(image_paths[0])
    target_image_folder = os.path.dirname(image_draw_path)
    if os.path.exists(target_image_folder):
        shutil.rmtree(target_image_folder)
    os.makedirs(target_image_folder)
    
    for image_idx, image_path in enumerate(tqdm(image_paths), start=1):
        image_handled = False
        image_idx_str = str(image_idx + 4)
        if image_idx < 390:
            continue
        if image_idx_str in markup_data:
            for obj1_id, markup_obj1 in enumerate(markup_data[image_idx_str], start=1):
                for obj2_id, markup_obj2 in enumerate(markup_data[image_idx_str], start=1):
                    if obj2_id <= obj1_id:
                        continue
                    if markup_obj1["class_name"] in ignore_objects or \
                            markup_obj2["class_name"] in ignore_objects:
                        continue
                    print_and_log(f'Handling {image_path} obj1: {obj1_id}, obj2: {obj2_id}')
                    coords1 = [
                        markup_obj1["x1"], markup_obj1["y1"],
                        markup_obj1["x2"], markup_obj1["y2"]
                    ]
                    coords2 = [
                        markup_obj2["x1"], markup_obj2["y1"],
                        markup_obj2["x2"], markup_obj2["y2"]
                    ]
                    with Image.open(image_path) as image:
                        image_size = image.size

                    coords1_model = convert_coords(coords1, image_size, to_model_format=True)
                    coords2_model = convert_coords(coords2, image_size, to_model_format=True)

                    # prompt_text = f'What\'s the relation between ' \
                    #               f'a {markup_obj1["class_name"]} {coords1_model} {DEFAULT_REGION_FEA_TOKEN} and ' \
                    #               f'a {markup_obj2["class_name"]} {coords2_model} {DEFAULT_REGION_FEA_TOKEN}? ' \
                    #               f'Choose one from "on", "next to", "in", "has"'

                    prompt_text = f'What is the physical relation between ' \
                                  f'{markup_obj1["class_name"]} {coords1_model} {DEFAULT_REGION_FEA_TOKEN} and ' \
                                  f'{markup_obj2["class_name"]} {coords2_model} {DEFAULT_REGION_FEA_TOKEN}?'

                    # prompt_text = f'Can you figure out the geometric relation of the ' \
                    #               f'{markup_obj1["class_name"]} {coords1_model} {DEFAULT_REGION_FEA_TOKEN} and ' \
                    #               f'{markup_obj2["class_name"]} {coords2_model} {DEFAULT_REGION_FEA_TOKEN}?'

                    
                    image_draw_path = prepare_out_path(image_path, f'{obj1_id}_{obj2_id}')
                    
                    image2draw = cv2.imread(image_path)
                    draw_frame_id(image2draw, image_idx)

                    masks = [
                        generate_mask_for_feature(coords1_model, raw_w=IMAGE_W, raw_h=IMAGE_H),
                        generate_mask_for_feature(coords2_model, raw_w=IMAGE_W, raw_h=IMAGE_H)
                    ]
                    
                    output_message = send_promt_with_image(
                        image_path, prompt_text, masks
                    )
                    position_map = {
                        'under': 'under', 
                        'far away': 'far away', 
                        'next to': 'next to',
                        'has': 'has',
                        'behind': 'behind',
                        'in': 'in',
                        'on': 'on'
                    }
                    
                    bboxes = get_coords_from_text(output_message, image_size)
                    draw_boxes(image2draw, bboxes)

                    position_label = get_position_from_text(output_message, position_map)

                    if position_label is None:
                        color = (0, 0, 228)
                        position_label = 'Unknown relation'
                    color = (50, 50, 220)
                    # image2draw = cv2.resize(image2draw, (1000, 1000))
                    draw_relation(image2draw, coords1, coords2, position_label, color)
                    # draw_relation(image2draw, coords1_model, coords2_model, position_label, color)

                    cv2.imwrite(image_draw_path, image2draw)
                    image_handled = True
                    # return
    
        if not image_handled:
            image_draw_path = prepare_out_path(image_path)
            cv2.imwrite(image_draw_path, cv2.imread(image_path))
        print_and_log('')


def describe_images(image_paths, prompt_text, use_ferret, log_status=False, replace_prompt=False):
    image_draw_path = prepare_out_path(image_paths[0])
    target_image_folder = os.path.dirname(image_draw_path)
    if os.path.exists(target_image_folder):
        shutil.rmtree(target_image_folder)
    os.makedirs(target_image_folder)
    
    PROCESS_EVERY_N = 10
    if log_status:
        data2iterate = tqdm(image_paths)
    else:
        data2iterate = image_paths

    generated_descriptions = []
    for image_idx, image_path in enumerate(data2iterate, start=1):
        # if image_idx % PROCESS_EVERY_N != 0:
        #     continue
        if log_status:
            print_and_log(f'Describing {image_path}')
        
        if use_ferret:
            with Image.open(image_path) as image:
                image_size = image.size

            image_draw_path = prepare_out_path(image_path, f'decs')
            
            image2draw = cv2.imread(image_path)
            draw_frame_id(image2draw, image_idx)

            masks = []
            
            output_message = send_promt_with_image(
                image_path, prompt_text, masks, log_status, replace_prompt
            )
            
            bboxes = get_coords_from_text(output_message, image_size)
            draw_boxes(image2draw, bboxes)
            
            cv2.imwrite(image_draw_path, image2draw)
            if log_status:
                print_and_log('')
            
            generated_descriptions.append(output_message)
        else:
            image_features = load_images([image_path])
            image_sizes = [image.size for image in image_features]
            image_features = llava_chat.preprocess_image(image_features)

            outputs = llava_chat(query=prompt_text, image_features=image_features, image_sizes=image_sizes)
            if log_status:
                print_and_log(outputs)
                print_and_log('')
            
            generated_descriptions.append(outputs)

    return generated_descriptions

def read_json_markup(json_file):
    with open(json_file, 'r') as file:
        markup_data = json.load(file)
    
    return markup_data

def init_model(use_ferret):
    global LOG_FILE
    global llava_chat
    global worker_addr
    global model_name
    
    if LOG_FILE is not None:
        print("Model was already initialized")
        return
    
    if use_ferret:
        model_name = 'ferret-7b-v1-3'
        # model_name = 'ferret-13b-v1-3'
        controller_url = 'http://localhost:10000'

        ret = requests.post(controller_url + "/get_worker_address",
                json={"model": model_name})
        worker_addr = ret.json()["address"]
        print(f"model_name: {model_name}, worker_addr: {worker_addr}")

        LOG_FILE = 'ferret_bag3_descriptions_v2.txt'
    else:
        from llava_model import LLaVaChat
        LOG_FILE = 'llava_bag3_descriptions_v3.txt'
        model_path = "liuhaotian/llava-v1.6-vicuna-7b"
        llava_chat = LLaVaChat(model_path)


def main():
    parser = argparse.ArgumentParser(description='Process video or image for detection, relation, or description.')
    
    parser.add_argument('--image', type=str, default=None, help='Path to the image')
    parser.add_argument('--video', type=str, default=None, help='Path to the video')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt to use for the image/video')
    parser.add_argument('--detect', action='store_true', help='Enable detection mode')
    parser.add_argument('--relation', action='store_true', help='Enable relation mode')
    parser.add_argument('--description', action='store_true', help='Enable description mode')
    # parser.add_argument('--llava', action='store_true', help='Use LLaVa instead of ferret')

    args = parser.parse_args()

    assert (args.image and not args.video) or \
           (not args.image and args.video), \
           "Path to image/video must be provided"
    
    assert args.detect + args.relation + args.description == 1, \
           "Specify only one processing mode"

    # use_ferret = not args.llava
    use_ferret = True
    init_model(use_ferret)
    
    reset_log()

    if args.video:
        markup_path = 'markup/orig_bag3_all_objects.json'
        videopath = "/media/titrom/archive/mipt/ROS/docker_data/orig_bag3.mp4"
        image_paths = get_frames_from_video(videopath)
        
        markup_data = read_json_markup(markup_path)

        if args.detect:
            prompt_text = f'Where is the object {COORDS_TAG} located? ' \
                           f'Give the answer from the list: on the floor, on the table, on the chair'
            detect_objects(image_paths, markup_data, prompt_text)
        elif args.relation:
            detect_relations(image_paths, markup_data)
        elif args.description:
            description_prompt = f'Describe the image in details.'
            if args.prompt is None:
                args.prompt = description_prompt
            describe_images(image_paths, args.prompt, use_ferret, log_status=True)
        else:
            raise ValueError('Unexpected working mode, set DETECT_MODE or RELATION_MODE to True')
    else:
        image_path = args.image
        assert args.prompt
        
        prompt_text = args.prompt
        describe_images([image_path], args.prompt, use_ferret, log_status=True)


if __name__ == "__main__":
    main()