from vlm import VLM
from dataset import Task, SROIE, DocILE


#x = VLM("SmolVLM")
#response = x.predict("datasets/kie/images/0.png", "How much do I need to pay?")


for data in DocILE(task=Task.KIE, split="val"):
    print(data)