from tqdm import tqdm
import datasets
import csv
import os

dataset = datasets.load_dataset(
    "facebook/kilt_wikipedia",
    num_proc=os.cpu_count() // 2,
)["full"]
print("Total examples:", len(dataset))

with open("data/msmarco/kilt_wikipedia.csv", "w") as f:
    writer = csv.writer(f, delimiter="\t")
    cnt = 0
    for data in tqdm(dataset):
        text = "".join(data["text"]["paragraph"])
        for l in range(0, len(text), 512):
            # chunk
            writer.writerow([str(cnt), data["wikipedia_title"], text[l : l + 512]])
            cnt += 1
