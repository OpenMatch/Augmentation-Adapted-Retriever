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

task_len = {
    "high_school_biology": 32,
    "jurisprudence": 11,
    "prehistory": 35,
    "high_school_microeconomics": 26,
    "nutrition": 33,
    "high_school_geography": 22,
    "human_sexuality": 12,
    "astronomy": 16,
    "moral_scenarios": 100,
    "clinical_knowledge": 29,
    "electrical_engineering": 16,
    "econometrics": 12,
    "high_school_computer_science": 9,
    "college_biology": 16,
    "miscellaneous": 86,
    "high_school_mathematics": 29,
    "college_medicine": 22,
    "high_school_macroeconomics": 43,
    "us_foreign_policy": 11,
    "professional_law": 170,
    "high_school_government_and_politics": 21,
    "security_studies": 27,
    "public_relations": 12,
    "global_facts": 10,
    "marketing": 25,
    "high_school_chemistry": 22,
    "machine_learning": 11,
    "sociology": 22,
    "moral_disputes": 38,
    "college_physics": 11,
    "high_school_statistics": 23,
    "management": 11,
    "virology": 18,
    "high_school_physics": 17,
    "high_school_world_history": 26,
    "international_law": 13,
    "logical_fallacies": 18,
    "world_religions": 19,
    "professional_accounting": 31,
    "elementary_mathematics": 41,
    "conceptual_physics": 26,
    "college_computer_science": 11,
    "human_aging": 23,
    "high_school_psychology": 60,
    "college_mathematics": 11,
    "medical_genetics": 11,
    "abstract_algebra": 11,
    "professional_medicine": 31,
    "computer_security": 11,
    "philosophy": 34,
    "business_ethics": 11,
    "professional_psychology": 69,
    "high_school_us_history": 22,
    "high_school_european_history": 18,
    "college_chemistry": 8,
    "formal_logic": 14,
    "anatomy": 13,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, help="Task name")
    parser.add_argument("--subtask_name", type=str, help="Subtask name")
    parser.add_argument("--method_name", type=str, help="Method name")
    parser.add_argument("--score", type=str, help="Score")

    args = parser.parse_args()

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
    print("All: {:.1f}".format(np.mean(list(accs.values())) * 100))
    # print acc for each category
    sum = 0
    for category, subcategories in categories.items():
        print(
            "{}: {:.1f}".format(
                category, category_accs[category][0] / category_accs[category][1] * 100
            )
        )
        sum += category_accs[category][1]


if __name__ == "__main__":
    main()
