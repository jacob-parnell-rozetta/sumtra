import torch
import os
from datasets import load_dataset, DatasetDict
import argparse
from tqdm import tqdm
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pipeline.language_code_mapping import LANGUAGE_CODE_MAPPING
import logging


def log_writer(out_file, output_dict):
    """To log outputs"""
    if not os.path.exists(out_file):
        with open(out_file, "w") as f:
            json.dump(output_dict, f)
            f.write('\n')

    else:  # append
        with open(out_file, "a") as f:
            json.dump(output_dict, f)
            f.write('\n')


class BacktranslationDatasetGenerator:
    def __init__(self, dataset_name, lang, few_shot):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained('pretrained_lm/facebook-mbart-large-50-many-to-one-mmt').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('pretrained_lm/facebook-mbart-large-50-many-to-one-mmt', use_fast=False)
        self.dataset_name = dataset_name
        if few_shot:
            train_ds = load_dataset(f"scripts/utils/data_loaders/{args.dataset}_en_loader.py",
                                    f"english-{LANGUAGE_CODE_MAPPING[lang]['name']}",
                                    cache_dir='datasets/dataset_cache', split=[f'train[:{args.few_shot}]'])
            val_ds = load_dataset(f"scripts/utils/data_loaders/{args.dataset}_en_loader.py",
                                  f"english-{LANGUAGE_CODE_MAPPING[lang]['name']}",
                                  cache_dir='datasets/dataset_cache', split=[f'validation'])
            test_ds = load_dataset(f"scripts/utils/data_loaders/{args.dataset}_en_loader.py",
                                   f"english-{LANGUAGE_CODE_MAPPING[lang]['name']}",
                                   cache_dir='datasets/dataset_cache', split=[f'test'])
            self.dataset = DatasetDict({"train": train_ds[0], "validation": val_ds[0], "test": test_ds[0]})
        else:
            self.dataset = load_dataset(f"scripts/utils/data_loaders/{self.dataset_name}_en_loader.py",
                                        f"english-{LANGUAGE_CODE_MAPPING[lang]['name']}",
                                        cache_dir='datasets/dataset_cache')
        self.src_lang = 'en'
        self.tgt_lang = lang
        # target language is the source of the tokenizer
        if LANGUAGE_CODE_MAPPING[lang]['mbart_code'] is not None:
            self.tokenizer.src_lang = LANGUAGE_CODE_MAPPING[lang]['mbart_code']
        else:
            self.tokenizer.src_lang = LANGUAGE_CODE_MAPPING[lang]['alt_code']
        self.skip = []

    def extract_backtranslations(self):
        for k in self.dataset.keys():
            print(f"Running over: {k}")
            dataset_record_mapping = {"crosssum": ["text", "summary"], 'wikilingua': ['text', 'summary']}
            if self.dataset_name == 'crosssum':
                out_path = f"data/processed_data/{self.dataset_name}/english-{LANGUAGE_CODE_MAPPING[self.tgt_lang]['name']}_bt_{k}.jsonl"
            else:
                out_path = f"data/processed_data/{self.dataset_name}_bt/english-{LANGUAGE_CODE_MAPPING[self.tgt_lang]['name']}_bt_{k}.jsonl"
            record_prediction = dataset_record_mapping[self.dataset_name][1]  # want the summary here

            # Enumerate and extract keywords
            for i, record in enumerate(tqdm(self.dataset[k])):
                encoded_groundtruth = self.tokenizer(record[record_prediction], return_tensors="pt").to(self.device)
                generated_tokens = self.model.generate(**encoded_groundtruth)
                backtranslation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

                record['summary_backtranslation'] = backtranslation
                log_writer(out_path, record)


if __name__ == "__main__":
    logging.info(f"Current GPU device: {torch.cuda.current_device()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='crosssum')
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--few_shot", type=int, default=None)
    args = parser.parse_args()
    bdg = BacktranslationDatasetGenerator(dataset_name=args.dataset, lang=args.lang, few_shot=args.few_shot)
    bdg.extract_backtranslations()
