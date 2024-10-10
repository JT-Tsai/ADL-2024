import ipdb
import json
import matplotlib.pyplot as plt
import os
import numpy as np

mc_path = []
qa_path = []

path = "output"
for folder, subfolders, files in os.walk(path):
    for file in files:
        file_name = file.split(".")[0]
        if file_name == "metrics":
            type = folder.split("\\")[-1]
            if "mc" in type:
                mc_path.append(os.path.join(folder, file))
            else:
                qa_path.append(os.path.join(folder, file))

# print(mc_path)
print(qa_path)

mc_data = []
qa_data = []

for path in qa_path:
    with open(path, 'r') as file:
        data = json.load(file)
        qa_data.append(data)


print(len(qa_data))

os.makedirs("plot_result", exist_ok=True)
plt.figure(figsize = (10, 5))
for i, data in enumerate(qa_data):
    flag = False
    for key in data:
        if len(data[key]) >= 5:
            if key != "EM":
                flag = True
                x = np.arange(0, len(data[key]))
                plt.subplot(1, 2, 1)
                plt.plot(x, data[key], label = key)
                plt.xlabel("Epoch")
                plt.ylabel("loss")
                plt.legend()
            else:
                # flag = True
                plt.subplot(1, 2, 2)
                x = np.arange(0, len(data[key]))
                plt.plot(x, data[key], label = key)
                plt.xlabel("Epoch")
                plt.ylabel("EM")
                plt.legend()
    if flag:
        file_name = str(qa_path[i].split("\\")[1]) + '_{' + str(qa_path[i].split("\\")[2]) + '}'
        plt.suptitle(file_name)
        plt.tight_layout()
        
        plt.savefig(f"plot_result/{file_name}.png")
        plt.show()


