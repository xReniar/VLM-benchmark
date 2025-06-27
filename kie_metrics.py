import os
import json
from Levenshtein import distance as edit_distance
import time


gt_path = "datasets/kie/labels"
gt_labels = sorted(os.listdir(gt_path))


def get_kie_metrics(
    vlm_name: str
) -> float:
    responses_path = f"responses/parsed/kie/{vlm_name}"
    predictions = sorted(os.listdir(responses_path))

    edit_distances = []
    for (pred, gt) in zip(predictions, gt_labels):
        pred_json:dict = json.load(open(os.path.join(responses_path, pred), "r"))
        gt_json:dict = json.load(open(os.path.join(gt_path, gt), "r"))

        for gt_field in gt_json.keys():
            pred_value = pred_json.get(gt_field, "")

            pred_value = str(pred_value)
            gt_value = str(gt_json[gt_field])

            dist = edit_distance(pred_value, gt_value)
            max_len = max(len(pred_value), len(gt_value))
            if max_len == 0:
                edit_distances.append(1.0)
            else:
                edit_distances.append(1 - (dist / max_len))
    return round(sum(edit_distances) / len(edit_distances), 3)

'''
def smolvlm():
    responses_path = "responses/parsed/kie/smolvlm"
    predictions = sorted(os.listdir(responses_path))

    edit_distances = []
    for (pred, gt) in zip(predictions, gt_labels):
        pred_json:dict = json.load(open(os.path.join(responses_path, pred), "r"))
        gt_json:dict = json.load(open(os.path.join(gt_path, gt), "r"))

        for gt_field in gt_json.keys():
            pred_value = pred_json.get(gt_field, "")

            pred_value = str(pred_value)
            gt_value = str(gt_json[gt_field])

            dist = edit_distance(pred_value, gt_value)
            max_len = max(len(pred_value), len(gt_value))
            if max_len == 0:
                edit_distances.append(1.0)
            else:
                edit_distances.append(1 - (dist / max_len))
    return sum(edit_distances) / len(edit_distances)


def qwen2_5_VL():
    responses_path = "responses/parsed/kie/qwen2.5"
    predictions = sorted(os.listdir(responses_path))

    edit_distances = []
    for (pred, gt) in zip(predictions, gt_labels):
        pred_json:dict = json.load(open(os.path.join(responses_path, pred), "r"))
        gt_json:dict = json.load(open(os.path.join(gt_path, gt), "r"))

        for gt_field in gt_json.keys():
            pred_value = pred_json.get(gt_field, "")

            pred_value = str(pred_value)
            gt_value = str(gt_json[gt_field])

            dist = edit_distance(pred_value, gt_value)
            max_len = max(len(pred_value), len(gt_value))
            if max_len == 0:
                edit_distances.append(1.0)
            else:
                edit_distances.append(1 - (dist / max_len))
    return sum(edit_distances) / len(edit_distances)
'''


print(get_kie_metrics("smolvlm"))
print(get_kie_metrics("qwen2.5"))