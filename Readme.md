
# Model inference
python process_coco_markup.py MODEL_NAME MARKUP_PATH OUTPUT_PATH
## Example
python process_coco_markup.py kosmos coco_markup/scene0011_00
_coco/annotations_color_material.json coco_markup/test_kosmos_markup.json

# Metric calculation:
python metrics_calculation.py --prediction PATH_TO_RESULTS

## Example
python metrics_calculation.py --prediction coco_markup/scene0011_00_coco/annotations_color_material_ferret_llava_kosmos.json  
python metrics_calculation.py --prediction coco_markup/test_kosmos_markup.json