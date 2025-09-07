import os
import json
import matplotlib.pyplot as plt


def draw(dataset: str):
    qwen_log_r8 = json.load(open(f"r=8/qwen2.5-vl-normal-{dataset}.json", "r"))
    smol_log_r8 = json.load(open(f"r=8/smolvlm2-normal-{dataset}.json", "r"))

    qwen_log_r16 = json.load(open(f"r=16/qwen2.5-vl-normal-{dataset}.json", "r"))
    smol_log_r16 = json.load(open(f"r=16/smolvlm2-normal-{dataset}.json", "r"))
    
    slice = -1
    logs = [
        qwen_log_r8[:slice],
        qwen_log_r16[:slice],
        smol_log_r8[:slice],
        smol_log_r16[:slice]
    ]

    labels = ["Qwen2.5-VL (r=8)", "Qwen2.5-VL (r=16)", "SmolVLM2 (r=8)", "SmolVLM2 (r=16)"]

    
    train_losses = []
    eval_losses = []
    steps_list = []
    for log in logs:
        train_loss = []
        eval_loss = []
        steps = []
        for step_info in log:
            steps.append(step_info["step"])
            if "loss" in step_info:
                train_loss.append(step_info["loss"])
                steps_list.append(step_info["step"])
            if "eval_loss" in step_info:
                eval_loss.append(step_info["eval_loss"])

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        steps_list.append(steps)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Train Loss
    for losses, label in zip(train_losses, labels):
        axes[0].plot(range(1, len(losses)+1), losses, marker='o', label=label)
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title(f"Confronto train_loss tra modelli ({dataset})")
    axes[0].grid(True)
    axes[0].legend()

    # Eval Loss
    for losses, label in zip(eval_losses, labels):
        axes[1].plot(range(1, len(losses)+1), losses, marker='o', label=label)
    axes[1].set_xlabel("Epoche")
    axes[1].set_ylabel("Eval Loss")
    axes[1].set_title(f"Confronto eval_loss tra modelli ({dataset})")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()



draw("docile")