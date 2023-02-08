from sklearn.metrics import f1_score
import collections
import numpy as np
from generation_metrics import Metric
from t5.evaluation import metrics

def popqa(all_labels_real, all_preds_real):
    assert len(all_labels_real) == len(all_preds_real)
    accuracy = []
    for possible_answers, pred in zip(all_labels_real, all_preds_real):
        is_correct = False
        for pa in possible_answers:
            if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                is_correct = True
        accuracy.append(is_correct)
    return {"accuracy": 100 * np.mean(accuracy)}

T0_METRICS = {
    "BLEU": metrics.bleu,
    "ROUGE": metrics.rouge,
    "Span Squad": metrics.span_squad,
    "Squad": metrics.squad,
    "Trivia QA": metrics.trivia_qa,
    "Accuracy": metrics.accuracy,
    "Spearman Correlation": metrics.spearman_corrcoef,
    "Other": metrics.accuracy,
    "popQA": popqa
}

def _metric_max_over_ground_truths(metric_fn, ground_truths, prediction):
  """Computes the maximum of the metric over all ground truths."""
  return max(
      [metric_fn(ground_truth, prediction) for ground_truth in ground_truths]
  )


def _str_em(target, prediction):
  return target == prediction


def _str_f1(target, prediction):
  """Computes token f1 score for a single target and prediction."""
  prediction_tokens = prediction.split()
  target_tokens = target.split()
  common = (collections.Counter(prediction_tokens) &
            collections.Counter(target_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(target_tokens)
  f1 = 100 * (2 * precision * recall) / (precision + recall)
  return f1


def acc_single_ref(all_preds, all_labels):
    '''
    all_preds: List[Int]
    all_labels: List[Int]
    '''
    acc = 100 * np.mean([int(p == l) for p, l in zip(all_preds, all_labels)])
    return {"acc": acc}


def acc_multi_ref(all_preds, all_labels):
    '''
    all_preds: List[Int]
    all_labels: List[List[Int]]
    '''
    acc = 100 * np.mean([int(any([p == ll for ll in l])) for p, l in zip(all_preds, all_labels)])
    return {"acc": acc}


def f1_single_ref(all_preds, all_labels):
    '''
    all_preds: List[Int]
    all_labels: List[Int]
    '''
    f1_macro = 100 * f1_score(all_labels, all_preds, average="macro")
    return {"f1": f1_macro}


def str_em_single_ref(all_preds, all_labels):
    '''
    all_preds: List[Int]
    all_labels: List[Int]
    '''    
    em = 100 * np.mean([_str_em(p, l) for p, l in zip(all_preds, all_labels)])
    return {"em": em}


def str_em_multi_ref(all_preds, all_labels):
    '''
    all_preds: List[Int]
    all_labels: List[List[Int]]
    '''    
    em = 100 * np.mean([_metric_max_over_ground_truths(_str_em, l, p) for p, l in zip(all_preds, all_labels)])
    return {"em": em}


def str_f1_single_ref(all_preds, all_labels):
    '''
    all_preds: List[Int]
    all_labels: List[Int]
    '''
    f1 = 100 * np.mean([_str_f1(p, l) for p, l in zip(all_preds, all_labels)])
    return {"f1": f1}


def str_f1_multi_ref(all_preds, all_labels):
    '''
    all_preds: List[Int]
    all_labels: List[List[Int]]
    '''
    f1 = 100 * np.mean([_metric_max_over_ground_truths(_str_f1, l, p) for p, l in zip(all_preds, all_labels)])
    return {"f1": f1}


def rouge_single_ref(all_preds, all_labels):
    metric = Metric()
    for l, p in zip(all_labels, all_preds):
        metric.forword([l.split()], p.split())
    
    metric_res, *_ = metric.close()

    return metric_res


def rouge_multi_ref(all_preds, all_labels):
    metric = Metric()
    for l, p in zip(all_labels, all_preds):
        metric.forword([ll.split() for ll in l], p.split())
    
    metric_res, *_ = metric.close()

    return metric_res

# Multi-rouge/multi-bleu. When there are multiple references, we want to get the
# rouge score that is highest. According to the authors, this is how it was done
# in the GEM paper.
# Source: https://github.com/google/BIG-bench/blob/main/bigbench/api/task_metrics.py
def rouge_fn(targets, predictions):
    """Computes ROUGE by taking the max ROUGE-N per reference and N."""
    # Following strategy from https://www.aclweb.org/anthology/W04-1013/.
    # Identify best reference per response and ROUGE type.
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    max_references = {rouge_type: [] for rouge_type in rouge_types}
    for targ_for_resp, resp in zip(targets, predictions):
        # Compute individual scores per example/ref pair.
        resp_scores = [metrics.rouge([t], [resp]) for t in targ_for_resp]
        # Find best scoring references for generated output and ROUGE type.
        for rouge_type in rouge_types:
            best_score_index = max(range(len(resp_scores)), key=lambda x: resp_scores[x][rouge_type])
            best_ref = targ_for_resp[best_score_index]
            # Add the reference to the new reference list.
            max_references[rouge_type].append(best_ref)
    # Compute metric for each of the reference lists for a ref type.
    results = {}
    for rouge_type in rouge_types:
        results[rouge_type] = metrics.rouge(max_references[rouge_type], predictions)[rouge_type]
        
    return results
  