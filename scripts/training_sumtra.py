# Script to train the joint T3L model
import os
import logging
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, MBartTokenizer, MBart50Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
from pytorch_lightning.profiler import AdvancedProfiler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pipeline.sum_plus_tra import CascadeModelForXLS
from pipeline.language_code_mapping import LANGUAGE_CODE_MAPPING
from pipeline.custom_utils import label_smoothed_nll_loss, log_writer, SummarizationDataset, load_data, metric_loader


class JoinTranslationTransferLearning(pl.LightningModule):
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

        # Load the model NMT plus LM
        self.model = CascadeModelForXLS(tokenizer=self.tokenizer,
                                        model_mt=self.args['model_trans_path'],  # translator
                                        model_ms=self.args['model_sum_path'],  # summarizer
                                        max_input_len=self.args['max_input_len'],  # input doc length
                                        max_output_len=self.args['max_output_len'],  # generated translation length
                                        freeze_strategy=self.args['freeze_strategy'],  # summarizer enc + translator dec
                                        t3l_arguments=self.args)  # pass all arguments in

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        if LANGUAGE_CODE_MAPPING[self.args['tgt']]['name'] != 'english' and \
                LANGUAGE_CODE_MAPPING[self.args['tgt']]['mrouge'] == 1:
            print("***** LOADING MROUGE + BERTSCORE")
            self.scorers = metric_loader(metrics=['mrouge', 'bertscore'],
                                         lang=LANGUAGE_CODE_MAPPING[self.args['tgt']]['name'])
        else:  # non support for mrouge OR english summaries
            print("***** LOADING ROUGE + BERTSCORE")
            self.scorers = metric_loader(metrics=["rouge", 'bertscore'])
        self.ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # Load right model for bertscore
        self.bertscore_model = 'facebook/mbart-large-50-one-to-many-mmt'

    def _prepare_mtlm_input(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        return input_ids, attention_mask

    def forward(self, src_ids, tgt_ids, src_attention_mask, tgt_attention_mask, backtranslation_ids, step_type):
        labels = tgt_ids[:, 1:].clone()
        decoder_input_ids = tgt_ids[:, :-1]

        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        outputs = self.model(
            src_ids,
            attention_mask=src_attention_mask,
            test_trans=self.test_trans_writer if False else None,
            decoder_input_ids=decoder_input_ids,  # to pass to forward of MT-LM
            decoder_attention_mask=decoder_attention_mask,  # to pass to forward of MT-LM
            backtranslation_ids=backtranslation_ids,  # pass the backtranslation ids for the MS-LM decoder
            scorer=[self.scorers, self.bertscore_model]
        )
        tra_logits, sum_logits, sum_pred_ids, sum_pred_embs = outputs

        if self.args['label_smoothing'] == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            assert tra_logits.shape[-1] == self.model.mtlm.config.vocab_size  # It has to be the same as the vocab size
            loss = self.ce_loss_fct(tra_logits.view(-1, tra_logits.shape[-1]), labels.view(-1))
            # loss = torch.tensor(0.0)
        else:
            lprobs = torch.nn.functional.log_softmax(tra_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args['label_smoothing'], ignore_index=self.tokenizer.pad_token_id
            )

        # Aux loss config
        if self.args['auxiliary_loss'] == 'teacher_forcing':
            bt_labels = backtranslation_ids[:, 1:].clone()
            if self.args['label_smoothing'] == 0:
                # Same behavior as modeling_bart.py, besides ignoring pad_token_id
                assert sum_logits.shape[-1] == self.model.mslm.config.vocab_size
                bt_loss = self.ce_loss_fct(sum_logits.view(-1, sum_logits.shape[-1]), bt_labels.view(-1))
            else:
                lprobs = torch.nn.functional.log_softmax(sum_logits, dim=-1)
                bt_loss, bt_nll_loss = label_smoothed_nll_loss(
                    lprobs, bt_labels, self.args['label_smoothing'], ignore_index=self.tokenizer.pad_token_id
                )
            combined_loss = self.args['auxiliary_lambdas'][0]*loss + self.args['auxiliary_lambdas'][1]*bt_loss
            return [combined_loss, loss, bt_loss, sum_pred_ids, sum_pred_embs]

        # In a normal scenario - no loss for the summarizer
        empty_bt_loss = torch.tensor(0.0).to('cuda:0')
        return [loss, loss, empty_bt_loss, sum_pred_ids, sum_pred_embs]

    def training_step(self, batch, batch_nb, step_type='train'):
        output = self.forward(*batch, step_type)
        loss = output[0]  # combined loss when using bt, only tra loss when no bt
        mtlm_loss = output[1]
        mslm_loss = output[2]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss.detach(), 'lr': lr,
                            'mtlm_loss': mtlm_loss.detach(), 'mslm_loss': mslm_loss.detach(),
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(
                                loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0,
                            }
        self.log("my_lr", lr, prog_bar=True, on_step=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb, step_type='val'):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch, step_type)
        vloss = outputs[0]  # combined loss when using bt, tra loss only when not using bt
        summarization_preds = outputs[3]
        summarization_embs = outputs[4]

        # soft inference
        # pass embeds into mtlm.encoder() -> extract encoder outputs and pass to generate in model_kwargs.
        # tran_encoder_outputs = self.model.mtlm.model.encoder(inputs_embeds=summarization_embs, output_attentions=True, return_dict=True)
        # encoder_outputs = {'encoder_outputs': tran_encoder_outputs}
        # generated_ids = self.model.mtlm.generate(forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lan_code],
        #                                          max_length=self.args['max_output_len'],
        #                                          num_beams=1, num_beam_groups=1, do_sample=False,
        #                                          **encoder_outputs)
        # Generate translation
        src_ids, tgt_ids, src_attention_mask, tgt_attention_mask, bt_ids = batch
        # Recreate MT-LM attention mask based on MS-LM predictions
        summarization_preds, src_attention_mask = self._prepare_mtlm_input(summarization_preds)
        generated_ids = self.model.mtlm.generate(input_ids=summarization_preds, attention_mask=src_attention_mask,
                                                 forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lan_code],
                                                 max_length=self.args['max_output_len'],
                                                 num_beams=1, num_beam_groups=1, do_sample=False)

        with self.tokenizer.as_target_tokenizer():  # decode prediction to target language
            generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        gold_str = self.tokenizer.batch_decode(tgt_ids.tolist(), skip_special_tokens=True)
        bt_gold_str = self.tokenizer.batch_decode(bt_ids.tolist(), skip_special_tokens=True)

        # Write to file - for individual metrics
        out_file = f"{self.args['save_dir']}/model_preds_{step_type}.jsonl"
        output_r1 = output_r2 = output_rl = output_bs = 0.0
        # Batch decode intermediate predictions
        int_preds = self.tokenizer.batch_decode(summarization_preds.tolist(), skip_special_tokens=True)
        for ref, pred, intermediate, bt_ref in zip(gold_str, generated_str, int_preds, bt_gold_str):
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
            output_bs += bert_score['f1'][0]

            json_dict = {
                "epoch": f"val_{self.current_epoch}" if step_type == 'val' else 'test',
                "target": ref,
                "prediction": pred,
                "intermediate_summary": intermediate if self.args['int_summary'] else None,
                "bt_ground_truth": bt_ref if self.args['auxiliary_loss'] else None,
                "rouge-1": round(rouge1*100, 2), "rouge-2": round(rouge2*100, 2), "rouge-L": round(rougel*100, 2),
                "bertscore": bert_score['f1'][0]
            }
            log_writer(out_file, json_dict)
        output_r1 /= len(generated_str)
        output_r2 /= len(generated_str)
        output_rl /= len(generated_str)
        output_bs /= len(generated_str)

        self.log("val_loss", vloss, prog_bar=True, on_step=True)

        return {'vloss': vloss,
                'rouge1': vloss.new_zeros(1) + output_r1,
                'rouge2': vloss.new_zeros(1) + output_r2,
                'rougeL': vloss.new_zeros(1) + output_rl,
                'bertscore': vloss.new_zeros(1) + output_bs,
                'mtlm_loss': vloss.new_zeros(1) + outputs[1],
                'mslm_loss': vloss.new_zeros(1) + outputs[2]
                }

    def validation_epoch_end(self, outputs, step_type='val'):
        for name, p in self.model.named_parameters():
            p.requires_grad = True
        # Freeze params according to freeze strategy
        self.model.freeze_params()
        names = ['vloss', 'rouge1', 'rouge2', 'rougeL', 'bertscore', 'mtlm_loss', 'mslm_loss']
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
        self.log("mtlm_loss", logs['mtlm_loss'], prog_bar=True)
        self.log("mslm_loss", logs['mslm_loss'], prog_bar=True)

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
        # scheduler = get_constant_schedule(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        dataset = SummarizationDataset(hf_dataset=self.hf_datasets[split_name], tokenizer=self.tokenizer,
                                       training_arguments=self.args)
        sampler = None
        # Shuffle or not
        is_shuffle = True if is_train and (sampler is None) else False
        if is_train:
            return DataLoader(dataset, batch_size=self.args['batch_size'], shuffle=is_shuffle,
                              num_workers=self.args['num_workers'], sampler=sampler, pin_memory=True,
                              collate_fn=SummarizationDataset.collate_fn)
        else:  # for validation and inference
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
        parser.add_argument("--src", type=str, required=True, help='Source language.')
        parser.add_argument("--tgt", type=str, required=True, help='Target language.')
        parser.add_argument("--save_dir", type=str, default='cross_lingual_summarization')
        parser.add_argument("--save_prefix", type=str, default='test')
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size")
        parser.add_argument("--grad_accum", type=int, default=8, help="number of gradient accumulation steps")
        parser.add_argument("--gpus", type=int, default=1, help="Number of gpus. 0 for CPU")
        parser.add_argument("--warmup", type=int, default=0, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
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
        parser.add_argument("--model_sum_path", type=str,
                            default='pretrained_lm/facebook-mbart-large-50-many-to-one-mmt',
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--model_trans_path", type=str,
                            default='pretrained_lm/facebook-mbart-large-50-one-to-many-mmt',
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer", type=str, default='pretrained_lm/facebook-mbart-large-50-many-to-one-mmt',
                            help="The tokenizer is set to the same as the summarizer by default")
        parser.add_argument("--progress_bar", type=int, default=10, help="Progress bar. Good for printing")
        parser.add_argument("--precision", type=int, default=32, help="Double precision (64), full precision (32) "
                                                                      "or half precision (16). Can be used on CPU, "
                                                                      "GPU or TPUs.")
        parser.add_argument("--freeze_strategy", type=str,
                            choices=['train_1-2', 'train_3-4', 'train_2-3', 'train_all',
                                     'train_2', 'train_1', 'train_4', 'train_3',
                                     'train_2-4', 'train_1-3', 'train_1-4',
                                     'train_2-3-4', 'train_1-3-4', 'train_1-2-4', 'train_1-2-3'],
                            default='train_all', help="Modules fixing strategy.")
        parser.add_argument("--amp_backend", type=str, default='apex', help="The mixed precision backend to "
                                                                            "use ('native' or 'apex')")
        parser.add_argument("--weight_decay", type=float, default=0.01, help="Adam weight decay")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--from_pretrained", type=str, default=None,
                            help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--adafactor", action='store_true', help="Use adafactor optimizer")
        parser.add_argument("--dataset", type=str, default="xsum", help="Custom dataset choice")
        parser.add_argument("--dataset_cache", type=str, default='datasets/dataset_cache')
        parser.add_argument("--few_shot", type=int, default=-1, help="Num examples for few-shot training.")
        parser.add_argument("--int_summary", action='store_true', help="Intermediate summaries generator")
        parser.add_argument("--tau", type=float, default=1.0, help="Softmax temperature")
        parser.add_argument("--auxiliary_loss", type=str, default='teacher_forcing',
                            choices=[None, 'teacher_forcing'],
                            help="Additional back-translation objective")
        parser.add_argument("--auxiliary_lambdas", type=float, nargs="+", default=[0.01, 0.99],
                            help="lambdas to weight the combined loss functions - first applies to Tra loss, second"
                                 "applies to Sum loss")
        parser.add_argument("--gumbel_softmax", action='store_true',
                            help="apply gumbel softmax instead of normal softmax")
        parser.add_argument("--use_hard_predictions", action='store_true',
                            help="use hard predictions instead of soft embeddings in the model coupling")
        parser.add_argument("--profiler", action='store_true', help="Run the profiler")

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
        model = JoinTranslationTransferLearning.load_from_checkpoint(args.from_pretrained)
    else:
        model = JoinTranslationTransferLearning(vars(args))

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
        monitor='bertscore',
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

    trainer = pl.Trainer(gpus=args.gpus, accelerator='ddp', num_nodes=1,
                         # distributed_backend='ddp' if torch.cuda.is_available() else None,
                         profiler=profiler if args.profiler else None,
                         track_grad_norm=-1,
                         max_epochs=args.epochs if not args.debug else 100,
                         max_steps=-1,  # None if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.grad_accum,
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
        # trainer.test(model)
    else:
        # If doing zero-shot, the model we pass in the checkpoint is loaded and tested
        trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="cross_lingual_summarization")
    parser = JoinTranslationTransferLearning.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
