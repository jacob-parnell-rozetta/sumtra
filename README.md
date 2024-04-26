# SumTra:  Differentiable Pipeline for Few-Shot Cross-Lingual Summarization
### Paper accepted to appear at NAACL 2024

The code repository for the XLS research paper with the same title.

## Installation

Python version > 3.6.

Install required packages with pip:

python
```
pip install -r requirements.txt

# Install our adapted transformers package
cd pipeline/transformers
pip install -e .
cd ../..
```

## Data
### CrossSum
The En-X split of CrossSum is downloaded from [here](https://drive.google.com/file/d/11yCJxK5necOyZBxcJ6jncdCFgNxrsl4m/view?pli=1).
The dataset is also available on HuggingFace datasets. We use the HF dataset loader to generate the few-shot splits.

### WikiLingua
The En-X split of WikiLingua is downloaded from [here](https://drive.google.com/file/d/1PM7GFCy2gJL1WHqQz1dzqIDIEN6kfRoi/view).
We process these files into .jsonl files for ingestion via the HF dataset loader, which is also used to generate the
few-shot splits.

## Download Models
Download the pre-trained language models locally to run the scripts. We run PyTorch Lightning such that the models are loaded
from a local path. To do this, edit the arguments in the ```scripts/download_huggingface_model.py``` script and run.

## Getting started
Examples on how to train mBART-50 on monolingual data, and then leverage the same trained model
in SumTra for fine-tuning over the English-Spanish split of CrossSum.
### Monolingual Training of Sum
```bash
python3 scripts/train_summarizer.py --dataset='crosssum' \
                                    --tokenizer="pretrained_lm/facebook-mbart-large-50-many-to-one-mmt" \
                                    --model_lm_path="pretrained_lm/facebook-mbart-large-50-many-to-one-mmt" \
                                    --max_output_len=84 \
                                    --src="en" --tgt="en" \
                                    --save_dir="outputs/english_spanish_monolingual/1"
```
Once you have obtained a well-trained summarizer that can perform monolingual summarization, you can load this model checkpoint
alongside a pre-trained mBART translator and fine-tune the SumTra network altogether.
### Fine-Tune SumTra
```bash
python3 scripts/training_sumtra.py --int_summary \
                                   --auxiliary_lambdas 0.01 0.99 --auxiliary_loss='teacher_forcing' \
                                   --freeze_strategy="train_all" \
                                   --max_output_len=84 --few_shot=100 \
                                   --dataset="crosssum" \
                                   --tokenizer="pretrained_lm/facebook-mbart-large-50-many-to-one-mmt" \
                                   --model_trans_path="pretrained_lm/facebook-mbart-large-50-one-to-many-mmt" \
                                   --model_sum_path=$PATH_TO_PRETRAINED_SUM \
                                   --epochs=10 --src="en" --tgt="es" \
                                   --save_dir="outputs/english_spanish_sumtra/1"
```

### Model Testing / Zero-Shot Inference

To perform inference (test) over the test sets with the trained
models, simply add the `--test` flag and the `--from_pretrained` argument
providing the path to the trained checkpoint.

## Citation

Please cite our paper in your work:
### Note: Paper accepted to appear at NAACL 2024. Pre-print available in ArXiv.

```text
@misc{parnell2024sumtra,
      title={SumTra: A Differentiable Pipeline for Few-Shot Cross-Lingual Summarization}, 
      author={Jacob Parnell and Inigo Jauregi Unanue and Massimo Piccardi},
      year={2024},
      eprint={2403.13240},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
