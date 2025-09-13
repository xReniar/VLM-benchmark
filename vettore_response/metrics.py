import os
import json
from Levenshtein import distance as edit_distance


def metrics(fn_path: str, gt_path: str):
    predictions = json.load(open(fn_path, "r"))

    edit_distances = []

    for img_fn in predictions.keys():
        prediction: dict = predictions[img_fn]
        response: dict = prediction["response"]

        gt_fn: str = img_fn.split(".")[0]
        gt_json: dict = json.load(open(os.path.join(gt_path, f"{gt_fn}.json"), "r"))
        
        for gt_field in gt_json.keys():
            pred_value = response.get(gt_field, "")

            pred_value = str(pred_value)
            gt_value = str(gt_json[gt_field])

            dist = edit_distance(pred_value, gt_value)
            max_len = max(len(pred_value), len(gt_value))
            if max_len == 0:
                edit_distances.append(1.0)
            else:
                edit_distances.append(1 - (dist / max_len))

    return round(sum(edit_distances) / len(edit_distances), 3)

def single_metrics(fn_path: str, gt_path: str, dataset):
    predictions = json.load(open(fn_path, "r"))

    metrics_json = {}
    for img_fn in predictions.keys():
        current_metrics = {}
        prediction: dict = predictions[img_fn]
        response: dict = prediction["response"]

        gt_fn: str = img_fn.split(".")[0]
        gt_json: dict = json.load(open(os.path.join(gt_path, f"{gt_fn}.json"), "r"))

        for gt_field in gt_json.keys():
            pred_value = response.get(gt_field, "")

            pred_value = str(pred_value)
            gt_value = str(gt_json[gt_field])

            dist = edit_distance(pred_value, gt_value)

            current_metrics[gt_field] = {
                "pred": pred_value,
                "gt": gt_value,
                "edit_distance": dist
            }

        metrics_json[img_fn] = current_metrics

    filename = fn.split("/")[-1]
    with open(f"single_metrics/{dataset}/{filename}.json", "w") as f:
        json.dump(metrics_json, f, indent=4)


for dataset in ["sroie", "docile"]:
    print(f"\n{dataset.upper()} dataset")
    gt_path = f"gt/{dataset}_gt"
    folder_path = f"result-{dataset}/parsed"
    for fn in sorted(os.listdir(folder_path)):
        print(f"{fn}: {metrics(f'{folder_path}/{fn}', gt_path)}")
        single_metrics(f'{folder_path}/{fn}', gt_path, dataset)