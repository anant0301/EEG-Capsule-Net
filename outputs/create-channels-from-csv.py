import numpy as np
import pandas as pd

df = pd.read_csv('./all-channels/all-ch-sub1-3class-training.csv', header= None)
l = []
for i in range(0, len(df.index), 30):
    if i+30 > len(df.index):
        break
    l.append(df.iloc[i:i+30].values.tolist())

arr = np.array(l)
ch_names = ["FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8","FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2","C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7","P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ","PO4","PO6","PO8","CB1","O1","OZ","O2","CB2"]
ch_dict = dict()
print(arr.mean())
exit()
for i, ch_name in enumerate(ch_names):
    ch_dict[ch_name.upper()] = i

import matplotlib.pyplot as plt


plt.figure(figsize=(20, 10))
for i, ind in enumerate(ch_names):
    if i >= 10:
        break
    plt.plot(arr[ch_dict[ind], :], label=ind)
plt.legend()
plt.title("EEG Channels: Testing Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Testing Accuracy")
plt.show()
