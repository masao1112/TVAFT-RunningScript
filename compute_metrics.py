import transformers, peft, accelerate, datasets, bitsandbytes, trl
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from tqdm.auto import tqdm
import evaluate


def main():
  #--------// Load responses từ các model
  dataset_name = "hungnm/vietnamese-medical-qa"
  full_dataset = load_dataset(dataset_name, split="train")
  train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42)
  raw_datasets = DatasetDict({
      'train': train_test_split['train'],
      'validation': train_test_split['test']
  })
  
  test_set = raw_datasets['validation']
  # Load standard answer
  std_answer = test_set['answer']
  
  # Load references
  base_res_df = pd.read_csv("base_model_responses_test")
  sft_res_df = pd.read_csv("sft_model_responses_test")
  nlft_res_df = pd.read_csv("nlft_model_responses_test")
  
  # convert from DataFrame -> List
  base_res_list = base_res_df['model_response'].values.tolist()
  sft_res_list = sft_res_df['model_response'].values.tolist()
  nlft_res_list = nlft_res_df['model_response'].values.tolist()
  
  # Compute metrics
  base_metrics = compute_all_metrics(base_res_list, std_answer)
  sft_metrics = compute_all_metrics(sft_res_list, std_answer)
  nlft_metrics = compute_all_metrics(nlft_res_list, std_answer)

  
  
# ================== Hàm tính metrics ==================
def compute_all_metrics(predictions, references):
      results = {}
      # BLEU
      bleu_metric = evaluate.load('bleu')
      references_for_bleu = [[ref] for ref in references]
      bleu_results = bleu_metric.compute(predictions=predictions, references=references_for_bleu)
      results['BLEU-4'] = bleu_results['bleu'] * 100

      # ROUGE
      rouge_metric = evaluate.load('rouge')
      rouge_results = rouge_metric.compute(predictions=predictions, references=references)
      results['ROUGE-1 (Recall)'] = rouge_results['rouge1'] * 100
      results['ROUGE-2 (Recall)'] = rouge_results['rouge2'] * 100
      results['ROUGE-L (Recall)'] = rouge_results['rougeL'] * 100

      # BERTScore
      bertscore_metric = evaluate.load("bertscore")
      bertscore_results = bertscore_metric.compute(predictions=predictions, references=references, lang="vi")
      results['BERTScore (F1)'] = np.mean(bertscore_results['f1']) * 100

      # Token-level Precision/Recall/F1
      def compute_token_metrics(predictions, references):
          f1_scores, precision_scores, recall_scores = [], [], []
          for pred, ref in zip(predictions, references):
              pred_tokens = set(pred.split())
              ref_tokens = set(ref.split())
              if not pred_tokens and not ref_tokens:
                  precision_scores.append(1.0); recall_scores.append(1.0); f1_scores.append(1.0)
                  continue
              if not pred_tokens or not ref_tokens:
                  precision_scores.append(0.0); recall_scores.append(0.0); f1_scores.append(0.0)
                  continue
              common_tokens = pred_tokens.intersection(ref_tokens)
              precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
              recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
              f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
              precision_scores.append(precision)
              recall_scores.append(recall)
              f1_scores.append(f1)
          return {
              "precision": np.mean(precision_scores) * 100,
              "recall": np.mean(recall_scores) * 100,
              "f1": np.mean(f1_scores) * 100
          }

      token_metrics = compute_token_metrics(predictions, references)
      results['Token-level Precision'] = token_metrics['precision']
      results['Token-level Recall'] = token_metrics['recall']
      results['Token-level F1 (Vietnamese)'] = token_metrics['f1']

      return results