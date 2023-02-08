from promptsource.templates import TemplateCollection
from tqdm import tqdm
import argparse
import openai
import json
import time
import numpy as np
import os

openai.api_key = "YOUR_API_KEY"
choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def eval(args, template, lines):
    cors = []
    all_probs = []
    answers = choices

    for line in tqdm(lines):
        sample = json.loads(line)
        applied_sample = template.apply(sample)
        prompt, label = applied_sample

        if args.method_name != "raw":
            prompt = (
                sample["passages"][0]
                + "\n\n"
                + sample["passages"][1]
                + "\n\n"
                + sample["passages"][2]
                + "\n\n"
                + prompt
            )

        # print("prompt:", prompt)
        # print("label:", label)

        while True:
            try:
                c = openai.Completion.create(
                    model=args.model_name,
                    prompt=prompt + "\n\nAnswer:",
                    max_tokens=1,
                    logprobs=100,
                    temperature=0,
                    echo=True,
                )
                # g = open("b.txt", "w")
                # g.write(json.dumps(c))
                break
            except:
                print("pausing")
                time.sleep(1)
                continue

        lprobs = []
        for ans in answers:
            try:
                lprobs.append(
                    c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)]
                )
            except:
                print(
                    "Warning: {} not found. Artificially adding log prob of -100.".format(
                        ans
                    )
                )
                lprobs.append(-100)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f}".format(acc))

    return cors, acc, all_probs


def main(args):
    collection = TemplateCollection()
    templates = collection.get_dataset("ai2_arc", "ARC-Challenge")
    template = templates["heres_a_problem"]
    cors, acc, _ = eval(args, template, open(args.data_dir).readlines())

    os.makedirs(
        args.save_dir
        + args.task_name
        + "/"
        + args.model_name
        + "/"
        + args.method_name
        + "/",
        exist_ok=True,
    )
    g = open(
        args.save_dir
        + args.task_name
        + "/"
        + args.model_name
        + "/"
        + args.method_name
        + "/"
        + "{:.3f}_0.txt".format(acc),
        "w",
    )
    for cor in cors:
        g.write(f"{cor}\n")


def genread(args):
    collection = TemplateCollection()
    templates = collection.get_dataset("ai2_arc", "ARC-Challenge")
    template = templates["heres_a_problem"]
    # genread_template = "Generate a background document from Wikipedia to answer the given question. {}"  # This prompt comes from the GenRead paper
    genread_template = "{} Generate a background document from Wikipedia to help answer the given question:"
    output_file = open("genread.jsonl", "a")

    for line in tqdm(open(args.data_dir).readlines()):
        sample = json.loads(line)
        applied_sample = template.apply(sample)
        prompt, _ = applied_sample
        prompt = genread_template.format(prompt)
        while True:
            try:
                c = openai.Completion.create(
                    model=args.model_name,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=150,
                    logprobs=5,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                break
            except:
                print("pausing")
                time.sleep(1)
                continue
        sample["passages"] = [c["choices"][0]["text"]]
        output_file.write(json.dumps(sample) + "\n")
        output_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="data/")
    parser.add_argument("--save_dir", "-s", type=str, default="results/")
    parser.add_argument("--task_name", type=str, default="mmlu_validation")
    parser.add_argument("--model_name", type=str, default="text-davinci-002")
    parser.add_argument("--method_name", type=str, default="raw")
    args = parser.parse_args()
    main(args)
    # genread(args)
