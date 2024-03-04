import json

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
# from cider import Cider

def calculate_metrics(reference, predicted):
    # Exact Match@1 (EM@1)
    em_1 = int(reference == predicted)
    
    # Exact Match@10 (EM@10)
    # em_10 = int(reference in predicted.split()[:10])

    reference_tokens = reference.split()
    predicted_tokens = predicted.split()
    
    # BLEU scores
    smoothing = SmoothingFunction().method1
    bleu_1 = sentence_bleu([reference_tokens], predicted_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = sentence_bleu([reference_tokens], predicted_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_3 = sentence_bleu([reference_tokens], predicted_tokens, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smoothing)
    bleu_4 = sentence_bleu([reference_tokens], predicted_tokens, smoothing_function=smoothing)
    
    # bleu_1 = sentence_bleu([reference_tokens], predicted_tokens, weights=(1, 0, 0, 0))
    # bleu_10 = sentence_bleu([reference_tokens], predicted_tokens, weights=(0, 0, 0, 1))

    # ROUGE score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predicted, reference)[0]
    rouge_score = rouge_scores['rouge-l']['f']
    
    # METEOR score
    meteor = meteor_score([reference_tokens], predicted_tokens)
    
    # CIDEr score
    # cider_scorer = Cider()
    # cider_score, _ = cider_scorer.compute_score({0: [reference]}, {0: [predicted]})
    
    return {
        'EM@1': em_1,
        # 'EM@10': em_10,
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-3': bleu_3,
        'BLEU-4': bleu_4,
        'ROUGE': rouge_score,
        'METEOR': meteor,
        # 'CIDEr': cider_score
    }


def calculate_ferret_metrics(markup_path, model):
    with open(markup_path, 'r') as json_file:
        scenes = json.load(json_file)

    metrics_dict = {}
    model_correct = 0
    object_count = 0

    for frame_id in scenes:
        scene_objects = scenes[frame_id]
        for obj_dict in scene_objects:
            obj_metrics = calculate_metrics(obj_dict["description"], obj_dict[f"{model}_description"])
            for k, v in obj_metrics.items():
                if k not in metrics_dict:
                    metrics_dict[k] = []
                metrics_dict[k].append(v)

            model_correct += obj_dict.get(f'{model}_correct', 0)
            object_count += 1

    print(f'{model} metrics:')
    for k, v in metrics_dict.items():
        metrics_dict[k] = round(sum(metrics_dict[k]) / len(metrics_dict[k]), 2)
    print(metrics_dict)
    if object_count > 0:
        print(f'Model correct answers: {round(model_correct / object_count, 2)}')

def main():
    # markup_path = "/media/titrom/archive/mipt/scanrefer_tester/coco_markup/scene0011_00_coco/annotations_validated.json"
    # markup_path = "/media/titrom/archive/mipt/scanrefer_tester/coco_markup/scene0011_00_coco/annotations_with_desc.json"
    # markup_path = "/media/titrom/archive/mipt/scanrefer_tester/coco_markup/scene0011_00_coco/annotations_with_desc_v3.json"
    # markup_path = "/media/titrom/archive/mipt/scanrefer_tester/coco_markup/scene0011_00_coco/desc_ferret_llava_v2.json"
    markup_path = "/media/titrom/archive/mipt/scanrefer_tester/coco_markup/scene0011_00_coco/desc_ferret_llava_kosmos_v2.json"
    calculate_ferret_metrics(markup_path, model="ferret")
    calculate_ferret_metrics(markup_path, model="llava")
    calculate_ferret_metrics(markup_path, model="kosmos")

if __name__ == "__main__":
    main()
