from vlm import VLM
from dataset import Task, SROIE, DocILE, MultiDataset


#x = VLM("Qwen2.5-VL")
#response = x.predict("data/kie/images/0.png", "How much do I need to pay?")

#print(response)

'''
for data in DocILE(tasks=[Task.KIE, Task.OCR], split="train"):
    print(data)
    image_path = data["image_path"]
    for entity in data["fields"]:
        print(entity)
'''

for data in DocILE(tasks=[Task.OCR, Task.KIE], split="train"):
    print(data)


'''
for data in MultiDataset([
    #(SROIE, [Task.OCR]),
    (DocILE, [Task.OCR])
], "train"):
    print(data)
'''