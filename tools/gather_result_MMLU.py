import argparse
import os
import csv
import numpy as np

subcategories_mapping = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
    ],
    "other (business, health, misc.)": ["other", "business", "health"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, help="Task name")
    parser.add_argument("--subtask_name", type=str, help="Subtask name")
    parser.add_argument("--method_name", type=str, help="Method name")
    parser.add_argument("--score", type=str, help="Score")

    args = parser.parse_args()

    dir_path = "/data/private/yuzc/Flan-T5-RA/data_hf/MMLU/val"
    task_len = {}
    sum = 0
    for file_name in os.listdir(dir_path):
        csv_path = os.path.join(dir_path, file_name)
        task_name = file_name[:-8]
        with open(csv_path, encoding="utf-8") as csv_file:
            task_len[task_name] = len(
                list(
                    csv.reader(
                        csv_file,
                        quotechar='"',
                        delimiter=",",
                        quoting=csv.QUOTE_ALL,
                        skipinitialspace=True,
                    )
                )
            )
            sum += task_len[task_name]
    # Hotfix
    task_len["anatomy"] -= 1

    accs = {}
    category_accs = {}
    with open(
        f"results/{args.method_name}/fp16/zs/{args.task_name}/preds/{args.task_name}/{args.score}_0.txt"
    ) as f:
        lines = f.readlines()[3:]
        begin, end = 0, 0
        for key, value in task_len.items():
            begin, end = end, end + value
            sum = 0
            for i in range(begin, end):
                pred, label = lines[i].strip().split("\t\t")
                sum += pred == label
            # gather acc for each category
            for category, subcategories in categories.items():
                for subcategory in subcategories:
                    if subcategory in subcategories_mapping[key]:
                        if category not in category_accs:
                            category_accs[category] = [0, 0]
                        category_accs[category][0] += sum / value
                        category_accs[category][1] += 1
            accs[key] = sum / value
    # print(accs)
    print(np.mean(list(accs.values())))
    # print acc for each category
    sum = 0
    for category, subcategories in categories.items():
        # print(
        #     f"{category}: {category_accs[category][0] / category_accs[category][1]}",
        #     end="\n",
        # )
        print(
            "{:.1f} &".format(
                category_accs[category][0] / category_accs[category][1] * 100
            ),
            end=" ",
        )
        sum += category_accs[category][1]
    # print(sum)


if __name__ == "__main__":
    main()
