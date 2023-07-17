# Script to train the language model for monolingual summarization
import os
import logging
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartTokenizer, MBart50Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pipeline.language_code_mapping import LANGUAGE_CODE_MAPPING
from pipeline.custom_utils import label_smoothed_nll_loss, log_writer, SummarizationDataset, load_data, metric_loader


class LmForSummarization(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.args = params
        self.hparams['params'] = params
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['tokenizer'], use_fast=False)
        print('Tokenizer Instance:   ', type(self.tokenizer))
        if isinstance(self.tokenizer, MBartTokenizer) or isinstance(self.tokenizer, MBart50Tokenizer):
            if LANGUAGE_CODE_MAPPING[self.args['src']]['mbart_code'] is not None:
                self.src_lan_code = LANGUAGE_CODE_MAPPING[self.args['src']]['mbart_code']
            else:
                self.src_lan_code = LANGUAGE_CODE_MAPPING[self.args['src']]['alt_code']
            if LANGUAGE_CODE_MAPPING[self.args['tgt']]['mbart_code'] is not None:
                self.tgt_lan_code = LANGUAGE_CODE_MAPPING[self.args['tgt']]['mbart_code']
            else:
                self.tgt_lan_code = LANGUAGE_CODE_MAPPING[self.args['tgt']]['alt_code']
            self.tokenizer.src_lang = self.src_lan_code
            self.tokenizer.tgt_lang = self.tgt_lan_code

        if LANGUAGE_CODE_MAPPING[self.args['tgt']]['name'] != 'english' and \
                LANGUAGE_CODE_MAPPING[self.args['tgt']]['mrouge'] == 1:
            print("***** LOADING MROUGE + BERTSCORE")
            self.scorers = metric_loader(metrics=['mrouge', 'bertscore'],
                                         lang=LANGUAGE_CODE_MAPPING[self.args['tgt']]['name'])
        else:  # non support for mrouge OR english summaries
            print("***** LOADING ROUGE + BERTSCORE")
            self.scorers = metric_loader(metrics=["rouge", 'bertscore'])

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args['model_lm_path'])
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

        self.ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # Load right model for bertscore
        self.bertscore_model = 'facebook/mbart-large-50-one-to-many-mmt'

    def forward(self, src_ids, tgt_ids, src_attention_mask, tgt_attention_mask, bt_ids, step_type):
        labels = tgt_ids[:, 1:].clone()
        decoder_input_ids = tgt_ids[:, :-1]

        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        outputs = self.model(
            src_ids,
            attention_mask=src_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False
        )
        lm_logits = outputs.logits
        if self.args['label_smoothing'] == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            assert lm_logits.shape[-1] == self.model.config.vocab_size  # It has to be the same as the output class
            loss = self.ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args['label_smoothing'], ignore_index=self.tokenizer.pad_token_id
            )

        # Metrics
        # acc = translation_metrics(lm_logits.detach(), labels, tokenizer=self.tokenizer)

        return [loss]

    def training_step(self, batch, batch_nb, step_type='train'):
        output = self.forward(*batch, step_type)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss.detach(), 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(
                                loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0,
                            }
                            #'accuracy': output[1]}
        self.log("my_lr", lr, prog_bar=True, on_step=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        # self.log("accuracy", output[1], prog_bar=True, on_step=True)
        # return {'loss': loss, 'accuracy': output[1], 'log': tensorboard_logs}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb, step_type='val'):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch, step_type)
        vloss = outputs[0]
        # Generate summary
        src_ids, tgt_ids, src_attention_mask, tgt_attention_mask, _ = batch
        # src_str = self.tokenizer.batch_decode(src_ids.tolist(), skip_special_tokens=True)
        if isinstance(self.tokenizer, MBartTokenizer) or isinstance(self.tokenizer, MBart50Tokenizer):
            generated_ids = self.model.generate(input_ids=src_ids, attention_mask=src_attention_mask,
                                                decoder_start_token_id=self.tokenizer.lang_code_to_id[self.tgt_lan_code],
                                                max_length=self.args['max_output_len'],
                                                num_beams=1, num_beam_groups=1, do_sample=False)
        else:  # mT5 is being used
            get_lang_id = lambda lang: self.tokenizer._convert_token_to_id(
                self.model.config.task_specific_params["langid_map"][lang][1]
            )
            generated_ids = self.model.generate(input_ids=src_ids, attention_mask=src_attention_mask,
                                                decoder_start_token_id=get_lang_id(LANG_CODE_MAP[self.args['tgt']]),
                                                max_length=self.args['max_output_len'],
                                                num_beams=1, num_beam_groups=1, do_sample=False)
        with self.tokenizer.as_target_tokenizer():
            generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        # generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        gold_str = self.tokenizer.batch_decode(tgt_ids.tolist(), skip_special_tokens=True)

        # Write to file - for individual metrics
        out_file = f"{self.args['save_dir']}/model_preds_{step_type}.jsonl"
        output_r1 = output_r2 = output_rl = output_bs = 0.0
        # output_bs = []
        for ref, pred in zip(gold_str, generated_str):
            rouge_score = self.scorers['rouge'].score(ref, pred)
            rouge1 = rouge_score['rouge1'].fmeasure
            rouge2 = rouge_score['rouge2'].fmeasure
            rougel = rouge_score['rougeL'].fmeasure
            output_r1 += rouge1
            output_r2 += rouge2
            output_rl += rougel
            # Embedding-based metrics
            bert_score = self.scorers['bertscore'].compute(references=[ref], predictions=[pred],
                                                           model_type=self.bertscore_model)
            # output_bs.append(bert_score['f1'][0])
            output_bs += bert_score['f1'][0]

            json_dict = {
                "epoch": f"val_{self.current_epoch}" if step_type == 'val' else 'test',
                "target": ref,
                "prediction": pred,
                "rouge-1": round(rouge1 * 100, 2), "rouge-2": round(rouge2 * 100, 2), "rouge-L": round(rougel * 100, 2),
                "bertscore": bert_score['f1'][0]
            }
            log_writer(out_file, json_dict)
        output_r1 /= len(generated_str)
        output_r2 /= len(generated_str)
        output_rl /= len(generated_str)
        # output_bs = np.average(np.array(output_bs))
        output_bs /= len(generated_str)

        self.log("val_loss", vloss, prog_bar=True, on_step=True)

        return {'vloss': vloss,
                'rouge1': vloss.new_zeros(1) + output_r1,
                'rouge2': vloss.new_zeros(1) + output_r2,
                'rougeL': vloss.new_zeros(1) + output_rl,
                'bertscore': vloss.new_zeros(1) + output_bs
                }

    def validation_epoch_end(self, outputs, step_type='val'):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ['vloss', 'rouge1', 'rouge2', 'rougeL', 'bertscore']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        self.log("validation_loss", logs['vloss'], prog_bar=True)
        self.log("rouge1", logs['rouge1'], prog_bar=True)
        self.log("rouge2", logs['rouge2'], prog_bar=True)
        self.log("rougeL", logs['rougeL'], prog_bar=True)
        self.log("bertscore", logs['bertscore'], prog_bar=True)

        return {'avg_val_loss': logs['vloss'], 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb, step_type='test')

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs, step_type='test')
        print(result)

    def configure_optimizers(self):
        if self.args['adafactor']:
            optimizer = Adafactor(self.model.parameters(), lr=self.args['lr'], scale_parameter=False, relative_step=False)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        if self.args['debug']:
            return optimizer  # const LR
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_steps = self.args['train_dataset_size'] * self.args['epochs'] / num_gpus / self.args['grad_accum'] / self.args['batch_size']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args['warmup'], num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        # add missing arguments from checkpoint
        if 'auxiliary_loss' not in self.args.keys():
            self.args['auxiliary_loss'] = None
        dataset = SummarizationDataset(hf_dataset=self.hf_datasets[split_name], tokenizer=self.tokenizer,
                                       training_arguments=self.args)
        sampler = None
        # Shuffle or not
        is_shuffle = True if is_train and (sampler is None) else False
        if is_train:
            return DataLoader(dataset, batch_size=self.args['batch_size'], shuffle=is_shuffle,
                              num_workers=self.args['num_workers'], sampler=sampler, pin_memory=True,
                              collate_fn=SummarizationDataset.collate_fn)
        else:
            return DataLoader(dataset, batch_size=self.args['val_batch_size'], shuffle=is_shuffle,
                              num_workers=self.args['num_workers'], sampler=sampler, pin_memory=True,
                              collate_fn=SummarizationDataset.collate_fn)

    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'validation', is_train=False)
        return self.val_dataloader_object

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--train_data", type=str, help='Path to training data')
        parser.add_argument("--validation_data", type=str, help='Path to validation data')
        parser.add_argument("--test_data", type=str, help='Path to testing data')
        parser.add_argument("--src", type=str, required=True, help='Source language.')
        parser.add_argument("--tgt", type=str, required=True, help='Target language.')
        parser.add_argument("--save_dir", type=str, default='summarization')
        parser.add_argument("--save_prefix", type=str, default='test')
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--val_batch_size", type=int, default=8, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=8, help="number of gradient accumulation steps")
        parser.add_argument("--max_grad_norm", type=float, default=1.0, help="number of gradient accumulation steps")
        parser.add_argument("--gpus", type=int, default=1,
                            help="Number of gpus. 0 for CPU")
        parser.add_argument("--warmup", type=int, default=500, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--weight_decay", type=float, default=0.01, help="Adam weight decay")
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--max_input_len", type=int, default=512,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_output_len", type=int, default=128,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--model_lm_path", type=str, default='../pretrained_lm/sshleifer-tiny-mbart',
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer", type=str, default='../pretrained_lm/sshleifer-tiny-mbart')
        parser.add_argument("--progress_bar", type=int, default=10, help="Progress bar. Good for printing")
        parser.add_argument("--precision", type=int, default=32, help="Double precision (64), full precision (32) "
                                                                      "or half precision (16). Can be used on CPU, "
                                                                      "GPU or TPUs.")
        parser.add_argument("--amp_backend", type=str, default='apex', help="The mixed precision backend to "
                                                                            "use ('native' or 'apex')")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--from_pretrained", type=str, default=None,
                            help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
        parser.add_argument("--attention_mode", type=str, default='sliding_chunks', help="Longformer attention mode")
        parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--adafactor", action='store_true', help="Use adafactor optimizer")
        parser.add_argument("--dataset", type=str, default="xsum", help="Custom dataset choice")
        parser.add_argument("--dataset_cache", type=str, default='datasets/dataset_cache')
        parser.add_argument("--few_shot", type=int, default=-1, help="Num examples for few-shot training. -1 for all.")
        parser.add_argument("--profiler", action='store_true', help="Run the profiler")
        parser.add_argument("--auxiliary_loss", default=None, help="This is never used for mBART m2m tests.")

        return parser


def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Current GPU device: {torch.cuda.current_device()}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.from_pretrained is not None:
        model = LmForSummarization.load_from_checkpoint(args.from_pretrained, vars(args))
    else:
        model = LmForSummarization(vars(args))

    # Load data
    model = load_data(model, args)
    logging.info(model.hf_datasets)

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        filename='check-{epoch:02d}-{bertscore:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='bertscore',  # monitor for highest rouge-2
        mode='max',
        every_n_epochs=1
    )
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    early_stopping = EarlyStopping(monitor='bertscore', patience=2,
                                   mode='max', check_on_train_epoch_end=True)
    callbacks = [checkpoint_callback, progress_bar, early_stopping]

    print(args)
    if args.profiler:
        from pytorch_lightning.profiler import AdvancedProfiler
        profiler = AdvancedProfiler(dirpath=f'{args.save_dir}/profiler/',
                                    filename=f'en{args.tgt}_profiler.txt')

    trainer = pl.Trainer(gpus=args.gpus, accelerator=None,
                         # distributed_backend='ddp' if torch.cuda.is_available() else None,
                         profiler=profiler if args.profiler else None,
                         track_grad_norm=-1,
                         max_epochs=args.epochs if not args.debug else 100,
                         max_steps=-1,  # None if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.grad_accum,
                         gradient_clip_val=args.max_grad_norm,
                         val_check_interval=args.val_every if not args.debug else 1,
                         num_sanity_val_steps=2,
                         check_val_every_n_epoch=1 if not args.debug else 1,
                         logger=logger,
                         callbacks=callbacks if not args.disable_checkpointing else False,
                         precision=args.precision,
                         amp_backend=args.amp_backend, amp_level='apex',
                         resume_from_checkpoint=args.resume_ckpt,
                         )
    # trainer.validate(model)
    if not args.test:
        trainer.fit(model)
        # After training or few shot tuning, the best checkpoint is run
        trainer.test(model, ckpt_path='best')
    else:
        # If doing zero-shot, the model we pass in the checkpoint is loaded and tested
        trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="summarization_model")
    parser = LmForSummarization.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
