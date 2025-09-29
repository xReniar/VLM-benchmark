import os
import json
import matplotlib.pyplot as plt


def draw(
    dataset: str,
    f1: str,
    f2: str,
    f3: str
):
    f1 = "smolvlm2-normal-r8.json.json"
    f2 = "smolvlm2-normal-r16.json.json"
    f3 = "smolvlm2-normal-24.json.json"

    f1_dict: dict = json.load(open(os.path.join(dataset, f1), "r"))
    f2_dict: dict = json.load(open(os.path.join(dataset, f2), "r"))
    f3_dict: dict = json.load(open(os.path.join(dataset, f3), "r"))

    
    freq_list: list[dict] = []
    for f_dict in [f1_dict, f2_dict, f3_dict]:
        frequency = {}
        for f_key in f_dict.keys():
            obj: dict = f1_dict[f_key]

            for field in obj.keys():
                #pred = field["pred"]
                #gt = field["gt"]
                edit_distance = obj[field]["edit_distance"]

                if edit_distance not in frequency:
                    frequency[edit_distance] = 1
                else:
                    frequency[edit_distance] += 1

        freq_list.append(frequency)

    for freq in freq_list:
        x_values = list(map(lambda x: x[0], list(freq.items())))[1:]
        y_values = list(map(lambda x: x[1], list(freq.items())))[1:]

        plt.bar(x_values, y_values, color="skyblue")
        plt.show()
        
        print(x_values)
        print(y_values)


draw(
    "sroie",
    "smolvlm2-normal-r8.json.json",
    "smolvlm2-normal-r16.json.json",
    "smolvlm2-normal-24.json.json"
)