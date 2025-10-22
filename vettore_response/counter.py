import os
import json
import time


dataset = "sroie"
folder_path = f"result-{dataset}/parsed"
f_before = "smolvlm2-normal-sroie-8.json"
f_after = f"smolvlm2-grpo-{dataset}.json"

before_dict = json.load(open(os.path.join(folder_path, f_before), "r"))
after_dict = json.load(open(os.path.join(folder_path, f_after), "r"))


f1_counter = 0
f2_counter = 0
for key in before_dict:
    obj = before_dict[key]
    before_response: dict = obj["response"]
    after_response: dict = after_dict[key]["response"]

    if before_response and after_response:
        if dataset == "sroie":
            pre_date: str = before_response["date"]
            after_date: str = before_response["date"]

            f1_counter += int(len(pre_date.split("/")) == 3)
            f2_counter += int(len(after_date.split("/")) == 3)

            #print(pre_date, after_date)

        if dataset == "docile":
            for fields in before_response.keys():
                if "date" in fields:
                    d1 = before_response[fields]
                    d2 = after_response[fields]

                    if d1 != d2:
                        print(d1, " ### ", d2)

print(f1_counter, f2_counter)