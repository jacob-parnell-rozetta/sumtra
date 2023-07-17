import os
import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict, load_metric
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from pipeline.language_code_mapping import LANGUAGE_CODE_MAPPING
from transformers import MBartTokenizer, MBart50Tokenizer


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


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class SummarizationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, training_arguments):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_input_len = training_arguments['max_input_len']
        self.max_output_len = training_arguments['max_output_len']
        # Add MS datasets here (XSum -> CrossSum)
        if training_arguments['dataset'] in ['xsum']:
            self.entry_input = 'document'
            self.entry_output = 'summary'
        elif training_arguments['dataset'] in ['cnn_dailymail']:
            self.entry_input = 'article'
            self.entry_output = 'highlights'
        else:  # Default to Cross-Lingual dataset loading (works for WikiLingua)
            self.entry_input = 'text'
            self.entry_output = 'summary'
            if training_arguments['auxiliary_loss']:
                self.intermediate_summary = 'summary_backtranslation'
            else:
                self.intermediate_summary = None

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        input_ids = self.tokenizer.encode(entry[self.entry_input], truncation=True, max_length=self.max_input_len)
        with self.tokenizer.as_target_tokenizer():  # to allow cross-lingual training
            output_ids = self.tokenizer.encode(entry[self.entry_output], truncation=True, max_length=self.max_output_len)
        src_attention_mask = [1] * len(input_ids)
        tgt_attention_mask = [1] * len(output_ids)

        if self.intermediate_summary:  # is not None
            backtranslation_ids = self.tokenizer.encode(entry[self.intermediate_summary], truncation=True,
                                                        max_length=self.max_output_len)

            return torch.tensor(input_ids), torch.tensor(output_ids),\
                   torch.tensor(src_attention_mask), torch.tensor(tgt_attention_mask), self.tokenizer,\
                   torch.tensor(backtranslation_ids)

        # this returns empty tensor for the backtranslations if we do not call it
        return torch.tensor(input_ids), torch.tensor(output_ids), \
               torch.tensor(src_attention_mask), torch.tensor(tgt_attention_mask), self.tokenizer, torch.zeros(1)

    @staticmethod
    def collate_fn(batch):
        # pad_token_id = 1  # For BART only (input and output IDs only)
        # pad_token_id = 0  # For mT5 only
        # Added dynamic pad_token_id depending on tokenizer used (mBART or mT5)
        input_ids, output_ids, src_attention_mask, tgt_attention_mask, tokenizer, backtranslation_ids = list(zip(*batch))
        if isinstance(tokenizer, MBartTokenizer) or isinstance(tokenizer, MBart50Tokenizer):
            pad_token_id = 1
        else:
            pad_token_id = 0
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        # Pad attention masks to 0
        src_attention_mask = torch.nn.utils.rnn.pad_sequence(src_attention_mask, batch_first=True, padding_value=0)
        tgt_attention_mask = torch.nn.utils.rnn.pad_sequence(tgt_attention_mask, batch_first=True, padding_value=0)

        if backtranslation_ids != torch.zeros(1):
            backtranslation_ids = torch.nn.utils.rnn.pad_sequence(backtranslation_ids, batch_first=True,
                                                                  padding_value=pad_token_id)
            return input_ids, output_ids, src_attention_mask, tgt_attention_mask, backtranslation_ids

        return input_ids, output_ids, src_attention_mask, tgt_attention_mask


def load_data(model, args):
    lang1 = LANGUAGE_CODE_MAPPING[args.src]['name']  # e.g. es -> spanish
    lang2 = LANGUAGE_CODE_MAPPING[args.tgt]['name']  # e.g. en -> english
    if args.few_shot > -1:  # Loading few_shot examples (if not -1 which denotes all)
        if args.auxiliary_loss:
            loader = f"scripts/utils/data_loaders/{args.dataset}_bt_loader.py"
        else:
            loader = f"scripts/utils/data_loaders/{args.dataset}_en_loader.py"

        train_ds = load_dataset(loader, f'{lang1}-{lang2}', cache_dir=args.dataset_cache,
                                split=[f'train[:{args.few_shot}]'])
        val_ds = load_dataset(loader, f'{lang1}-{lang2}', cache_dir=args.dataset_cache, split=[f'validation'])
        test_ds = load_dataset(loader, f'{lang1}-{lang2}', cache_dir=args.dataset_cache, split=[f'test'])
        model.hf_datasets = DatasetDict({"train": train_ds[0], "validation": val_ds[0], "test": test_ds[0]})

        print(model.hf_datasets)
    elif args.few_shot == -1:  # Loading full examples
        if args.dataset == 'xsum':
            model.hf_datasets = load_dataset("xsum", cache_dir=args.dataset_cache)
        elif args.dataset == 'cnn_dailymail':
            model.hf_datasets = load_dataset("cnn_dailymail", '3.0.0', cache_dir=args.dataset_cache)
        else:  # if args.dataset == 'crosssum': or wikilingua
            if args.auxiliary_loss:
                loader = f"scripts/utils/data_loaders/{args.dataset}_bt_loader.py"
            else:
                loader = f"scripts/utils/data_loaders/{args.dataset}_en_loader.py"
            model.hf_datasets = load_dataset(loader, f'{lang1}-{lang2}', cache_dir=args.dataset_cache)
    else:
        raise ValueError("Need to specify a dataset")

    # Add information for training
    args.train_dataset_size = model.hf_datasets['train'].num_rows  # for lr scheduler
    model.name_dataset = args.dataset
    return model


def metric_loader(metrics, lang=None):
    alt_languages_map = {'chinese_simplified': 'chinese', 'chinese_traditional': 'chinese', 'amharic': 'arabic',
                         'hausa': 'arabic', 'kyrgyz': 'turkish', 'oromo': 'arabic', 'punjabi': 'hindi',
                         'serbian_cyrillic': 'russian', 'somali': 'arabic', 'tigrinya': 'arabic', 'uzbek': 'turkish'}
    metrics_dict = {}
    for val in metrics:
        if val == 'mrouge':
            lang = alt_languages_map[lang] if lang in alt_languages_map.keys() else lang
            from scripts.utils.mrouge import rouge_scorer as mrouge_scorer
            rouge = mrouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=True,
                                              lang=lang)
            metrics_dict['rouge'] = rouge
        if val == 'rouge':
            rouge = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            metrics_dict['rouge'] = rouge
        if val == 'bleu':
            bleu_scorer = BLEU()
            metrics_dict['bleu'] = bleu_scorer
        if val == 'bertscore':
            # https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages
            bertscore = load_metric("bertscore")
            metrics_dict['bertscore'] = bertscore
    return metrics_dict

