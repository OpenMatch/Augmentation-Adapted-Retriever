# coding=utf-8


"""Training Enc-Dec"""

import os
import torch
import json
import numpy as np

from arguments import get_args
from data_utils.T0Datasets import T0Dataset
from data_utils.data_config import (
    DATA_GROUP_CONFIG,
    DATA_NO_EVAL,
    DATA_NO_VALID,
    DATA_NO_TRAIN,
    DATA_EVAL_GEN,
    DATA_RETRIEVAL_AUGMENTATION,
)
from data_utils import ANSWER_POST_FN
from tokenization_t5 import EncDecTokenizer

import mpu
from utils import save_checkpoint
from utils import print_args
from utils import print_rank_0, save_rank_0
from utils import save_preds_t0
from utils import setup_model_and_optimizer, set_random_seed, initialize_distributed

from samplers import DistributedBatchSampler, RandomSampler
from data_utils import *
from metrics import *

from torch.utils.data import DataLoader, SequentialSampler

from generation_utils import generate_beam, generate_no_beam
from promptsource.templates import TemplateCollection

from tqdm import tqdm


def forward_step(
    args,
    model_batch,
    no_model_batch,
    model,
    device,
    keep_enc_hidden=False,
    do_infer=False,
):
    for k in model_batch:
        model_batch[k] = model_batch[k].to(device)
    for k in no_model_batch:
        no_model_batch[k] = no_model_batch[k].to(device)

    if args.FiD:
        batch_size, _, sequence_length = model_batch["passage_input_ids"].size()
        enc_outputs = model(
            enc_input_ids=model_batch["passage_input_ids"].view(
                batch_size * args.passage_num, sequence_length
            ),
            enc_attention_mask=model_batch["passage_attention_mask"].view(
                batch_size * args.passage_num, 1, sequence_length, sequence_length
            ),
            only_encoder=True,
        )
        enc_hidden_states = enc_outputs["encoder_last_hidden_state"].view(
            batch_size, sequence_length * args.passage_num, -1
        )
        new_model_batch = {}
        for k in model_batch:
            if k not in ["passage_input_ids", "passage_attention_mask"]:
                new_model_batch[k] = model_batch[k]
        output = model(**new_model_batch, enc_hidden_states=enc_hidden_states)
    else:
        if keep_enc_hidden:
            enc_outputs = model(**model_batch, only_encoder=True)
            enc_hidden_states = enc_outputs["encoder_last_hidden_state"]
            output = model(**model_batch, enc_hidden_states=enc_hidden_states)
        else:
            output = model(**model_batch)

    logits = output["lm_logits"]
    forw_out = {"logits": logits}
    if keep_enc_hidden:
        forw_out["enc_hidden_states"] = enc_hidden_states

    if not do_infer:
        losses = mpu.vocab_parallel_cross_entropy(
            logits.contiguous().float(), no_model_batch["labels"]
        )
        loss_mask = no_model_batch["loss_mask"]
        losses = (losses * loss_mask).sum(-1) / loss_mask.sum(-1)
        loss = losses.mean()

        forw_out["loss"] = loss
        forw_out["loss_batch"] = losses

    return forw_out


def backward_step(args, loss, model, optimizer):
    # backward
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)


def train(
    args,
    data_names,
    tokenizer: EncDecTokenizer,
    model,
    optimizer,
    lr_scheduler,
    train_data_utils,
    dev_data_utils,
    device,
):
    """Train the model."""

    train_dataloader, train_dataset, random_sampler = train_data_utils

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss = 0.0

    step, global_step = 1, 1

    best_scores = []

    for e in range(args.epochs):
        model.train()
        random_sampler.set_epoch(e)
        train_dataset.set_epoch(e)
        for model_batch, no_model_batch, _, _ in train_dataloader:

            forw_out = forward_step(args, model_batch, no_model_batch, model, device)
            loss = forw_out["loss"]

            if torch.distributed.get_rank() == 0:
                print(loss)

            backward_step(args, loss, model, optimizer)

            # Update losses.
            total_loss += loss.item()

            if args.deepspeed:
                model.step()
            else:
                optimizer.step()
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()

            # Logging.
            if (
                global_step % args.log_interval == 0
                and step % args.gradient_accumulation_steps == 0
            ):
                learning_rate = optimizer.param_groups[0]["lr"]
                avg_lm_loss = total_loss / (
                    args.log_interval * args.gradient_accumulation_steps
                )
                log_string = "epoch {:3d}/{:3d} |".format(e, args.epochs)
                log_string += " global iteration {:8d}/{:8d} |".format(
                    global_step, args.train_iters
                )
                log_string += " learning rate {:.3} |".format(learning_rate)
                log_string += " lm loss {:.6} |".format(avg_lm_loss)
                if args.fp16:
                    log_string += " loss scale {:.1f} |".format(
                        optimizer.cur_scale if args.deepspeed else optimizer.loss_scale
                    )
                print_rank_0(log_string)
                save_rank_0(args, log_string)
                total_loss = 0.0

            # Checkpointing
            if (
                args.save
                and args.save_interval
                and global_step % args.save_interval == 0
                and step % args.gradient_accumulation_steps == 0
            ):
                save_dir_path = os.path.join(args.save, str(global_step))
                if torch.distributed.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                save_checkpoint(
                    global_step,
                    model,
                    optimizer,
                    lr_scheduler,
                    args,
                    save_dir=save_dir_path,
                )

            # Evaluation
            if (
                args.eval_interval
                and global_step % args.eval_interval == 0
                and step % args.gradient_accumulation_steps == 0
                and args.do_valid
            ):
                prefix = "iteration {} | ".format(global_step)
                metric_values = []
                for name, dev_data_util_prompt in dev_data_utils.items():
                    if DATA_CONFIG[name].get("selfsup", False):
                        if DATA_CONFIG[name]["type"] == "gen":
                            dev_dataloader, dev_dataset, _ = dev_data_util_prompt[0]
                            dev_loss = evaluate_lm(
                                args,
                                tokenizer,
                                name,
                                dev_dataset,
                                dev_dataloader,
                                model,
                                device,
                                mode="dev",
                                train_step=global_step,
                                save_res=True,
                            )
                            log_string = (
                                prefix + name + " | dev_loss: " + str(np.mean(dev_loss))
                            )
                            print_rank_0(log_string)
                            save_rank_0(args, log_string)
                        else:
                            dev_dataloader, dev_dataset, _ = dev_data_util_prompt[0]
                            dev_loss, dev_res, dev_preds, dev_labels = evaluate_rank(
                                args,
                                tokenizer,
                                name,
                                dev_dataset,
                                dev_dataloader,
                                model,
                                device,
                                mode="dev",
                                train_step=global_step,
                                save_res=True,
                            )
                            log_string = (
                                prefix
                                + name
                                + " | dev_loss: "
                                + str(np.mean(dev_loss))
                                + " | dev res: "
                                + str(dev_res)
                            )
                            print_rank_0(log_string)
                            save_rank_0(args, log_string)
                    else:
                        dev_res_prompt = []
                        dev_loss_prompt = []
                        dev_preds_prompt = []
                        dev_labels_prompt = []
                        dev_prompt_names = []
                        for pid, dev_data_util in enumerate(dev_data_util_prompt):
                            dev_dataloader, dev_dataset, _ = dev_data_util
                            dev_prompt_names.append(
                                dev_dataset.all_data[name]["prompt_names"][0]
                            )
                            if (
                                dev_dataset.data_prompts[name][0].answer_choices
                                is not None
                            ):
                                eval_func = evaluate_rank
                            else:
                                eval_func = evaluate_gen
                            dev_loss, dev_res, dev_preds, dev_labels = eval_func(
                                args,
                                tokenizer,
                                name,
                                dev_dataset,
                                dev_dataloader,
                                model,
                                device,
                                mode="dev",
                                train_step=global_step,
                                save_res=True,
                            )
                            dev_loss_prompt.append(dev_loss)
                            dev_res_prompt.append(dev_res)
                            dev_preds_prompt.append(dev_preds)
                            dev_labels_prompt.append(dev_labels)

                        log_string = (
                            prefix
                            + name
                            + " | dev_loss: "
                            + str(np.mean(dev_loss_prompt))
                            + " | dev res: "
                            + str(dev_res_prompt)
                        )

                        print_rank_0(log_string)
                        save_rank_0(args, log_string)
                        save_preds_t0(
                            args,
                            name,
                            dev_prompt_names,
                            global_step,
                            dev_res_prompt,
                            dev_preds_prompt,
                            dev_labels_prompt,
                        )

                        values = [
                            v for dev_res in dev_res_prompt for v in dev_res.values()
                        ]

                        metric_values.extend(values)

                if len(metric_values) != 0:
                    metric_avg = sum(metric_values) / len(metric_values)
                    log_string = prefix + "Average: " + str(metric_avg)
                    print_rank_0(log_string)
                    save_rank_0(args, log_string)
                model.train()

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

    return global_step


def evaluate_lm(
    args,
    tokenizer: EncDecTokenizer,
    name,
    eval_dataset: T0Dataset,
    eval_data_loader,
    model,
    device,
    mode="dev",
    train_step=0,
    save_res=False,
):
    model.eval()

    total_loss = 0.0
    step = 0

    with torch.no_grad():
        for model_batch, no_model_batch, _, _ in eval_data_loader:
            for k in model_batch:
                model_batch[k] = model_batch[k].to(device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(device)
            forw_out = forward_step(
                args, model_batch, no_model_batch, model, device, keep_enc_hidden=True
            )
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss
            step += 1

            if step == 0:
                if torch.distributed.get_rank() == 0:
                    print(name)
                    print(eval_dataset.data_prompts[name][0].name)
                    print(len(eval_dataset))

        total_loss /= step

    return total_loss


def evaluate_gen(
    args,
    tokenizer: EncDecTokenizer,
    name,
    eval_dataset: T0Dataset,
    eval_data_loader,
    model,
    device,
    mode="dev",
    train_step=0,
    save_res=False,
):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_output_ids = []
    all_idxs = []
    if args.FiD:
        all_scores = []

    with torch.no_grad():
        if not args.FiD:
            for model_batch, no_model_batch, _, _ in eval_data_loader:
                for k in model_batch:
                    model_batch[k] = model_batch[k].to(device)
                for k in no_model_batch:
                    no_model_batch[k] = no_model_batch[k].to(device)
                forw_out = forward_step(
                    args,
                    model_batch,
                    no_model_batch,
                    model,
                    device,
                    keep_enc_hidden=True,
                )
                loss = forw_out["loss"].item() if "loss" in forw_out else 0
                total_loss += loss
                step += 1

                if step == 0:
                    if torch.distributed.get_rank() == 0:
                        print(name)
                        print(eval_dataset.data_prompts[name][0].name)
                        print(len(eval_dataset))

            total_loss /= step

        for e, (model_batch, no_model_batch, _, _) in tqdm(
            enumerate(eval_data_loader), desc="Evaluating"
        ):
            for k in model_batch:
                model_batch[k] = model_batch[k].to(device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(device)
            if args.num_beams == 1:
                generation_str_list, generation_id_list = generate_no_beam(
                    model_batch,
                    model_batch["enc_input_ids"],
                    model,
                    tokenizer,
                    args,
                    device,
                )
                if args.FiD:
                    scores = model.module.module.module.get_crossattention_scores(
                        model_batch["passage_attention_mask"][:, :, 0, 0, :].bool()
                    )
                    all_scores.append(scores)
            else:
                generation_str_list, generation_id_list = generate_beam(
                    model_batch,
                    model_batch["enc_input_ids"],
                    model,
                    tokenizer,
                    args,
                    device,
                )

            output_ids = [
                x
                + [tokenizer.pad_id]
                + (args.max_generation_length - len(x)) * [tokenizer.pad_id]
                for x in generation_id_list
            ]
            output_ids = torch.tensor(output_ids).to(device)

            tmp_idxs = [
                torch.zeros_like(no_model_batch["idxs"]).to(device)
                for _ in range(mpu.get_data_parallel_world_size())
            ]
            torch.distributed.all_gather(
                tmp_idxs,
                no_model_batch["idxs"].data,
                group=mpu.get_data_parallel_group(),
            )

            tmp_output_ids = [
                torch.zeros_like(output_ids).to(device)
                for _ in range(mpu.get_data_parallel_world_size())
            ]
            torch.distributed.all_gather(
                tmp_output_ids, output_ids.data, group=mpu.get_data_parallel_group()
            )

            all_idxs.extend(tmp_idxs)
            all_output_ids.extend(tmp_output_ids)

    all_output_ids = torch.cat(all_output_ids, dim=0).cpu().tolist()
    all_idxs = torch.cat(all_idxs, dim=0).tolist()
    if args.FiD:
        all_scores = torch.cat(all_scores, dim=0)
        print(all_scores.size())
        torch.save(
            all_scores,
            os.path.join(args.save, f"stored_FiD_scores.pt"),
        )

    all_preds_real = []
    all_labels_real = []
    eval_res = {}
    for idxs, output_ids in zip(all_idxs, all_output_ids):
        _, _, sid = idxs
        output_ids = (
            output_ids[: output_ids.index(tokenizer.pad_id)]
            if tokenizer.pad_id in output_ids
            else output_ids
        )
        all_preds_real.append(tokenizer.decode(output_ids))
        all_labels_real.append(eval_dataset.all_data[name]["data"][sid]["answer"])

    metric_names = eval_dataset.data_prompts[name][0].metadata.metrics
    for metric_name in metric_names:
        if (name, metric_name) in ANSWER_POST_FN:
            all_labels_real, all_preds_real = ANSWER_POST_FN[(name, metric_name)](
                all_labels_real, all_preds_real
            )
        res = T0_METRICS[metric_name](all_labels_real, all_preds_real)
        eval_res.update(res)

    # if save_res:
    #     save_preds_t0(args, name, eval_dataset, train_step, eval_res, all_preds_real, all_labels_real)

    return total_loss, eval_res, all_preds_real, all_labels_real


def evaluate_rank(
    args,
    tokenizer: EncDecTokenizer,
    name,
    eval_dataset: T0Dataset,
    eval_data_loader,
    model,
    device,
    mode="dev",
    train_step=0,
    save_res=False,
):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_idxs = []
    all_preds = []
    if args.prompt_tune:
        all_prompt = torch.load(
            f"data_hf/{args.data_names}/cache/stored_dembeds.pt",
            map_location=lambda storage, loc: storage,
        )
    if args.FiD:
        all_scores = []

    tmp_pos_index = torch.arange(1, eval_dataset.max_cand_len + 1, device=device)
    with torch.no_grad():
        for (
            model_batch,
            no_model_batch,
            cand_model_batch,
            cand_no_model_batch,
        ) in tqdm(eval_data_loader, desc="Evaluating"):
            for k in model_batch:
                model_batch[k] = model_batch[k].to(device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(device)
            for k in cand_model_batch:
                cand_model_batch[k] = cand_model_batch[k].to(device)
            for k in cand_no_model_batch:
                cand_no_model_batch[k] = cand_no_model_batch[k].to(device)

            if args.prompt_tune:
                prompt = all_prompt[step]
                model.module.module.module.encoder.load_prompt_embeds(prompt)

            if args.FiD:
                model.module.module.module.reset_score_storage()
                batch_size, _, sequence_length = model_batch["passage_input_ids"].size()
                enc_outputs = model(
                    enc_input_ids=model_batch["passage_input_ids"].view(
                        batch_size * args.passage_num, sequence_length
                    ),
                    enc_attention_mask=model_batch["passage_attention_mask"].view(
                        batch_size * args.passage_num,
                        1,
                        sequence_length,
                        sequence_length,
                    ),
                    only_encoder=True,
                )
                enc_hidden_states = enc_outputs["encoder_last_hidden_state"].view(
                    batch_size, sequence_length * args.passage_num, -1
                )
            else:
                enc_outputs = model(**model_batch, only_encoder=True)
                enc_hidden_states = enc_outputs["encoder_last_hidden_state"]
                # enc_hidden_states[0, :10, :] = prompt
            output = model(**cand_model_batch, enc_hidden_states=enc_hidden_states)
            if args.FiD:
                scores = model.module.module.module.get_crossattention_scores(
                    model_batch["passage_attention_mask"][:, :, 0, 0, :].bool()
                )
                all_scores.append(scores)
            logits = output["lm_logits"]

            losses = mpu.vocab_parallel_cross_entropy(
                logits.contiguous().float(), cand_no_model_batch["target_ids"]
            )
            loss_mask = cand_no_model_batch["loss_mask"]

            losses = losses * loss_mask

            gold_loss = 0
            preds = []
            for samp_loss, cand_pos, cand_label in zip(
                losses, cand_no_model_batch["pos"], cand_no_model_batch["labels"]
            ):
                cum_loss = torch.cumsum(samp_loss, dim=0)
                # print(samp_loss)
                sum_loss = torch.masked_select(cum_loss, cand_pos)
                cand_loss = torch.diff(
                    sum_loss, dim=0, prepend=torch.zeros(1, device=device)
                )
                # print("cand loss", cand_loss)
                # print("samp loss", samp_loss)
                cand_pos_idx = torch.masked_select(tmp_pos_index, cand_pos)
                cand_lens = torch.diff(
                    cand_pos_idx, dim=0, prepend=torch.zeros(1, device=device)
                )
                # print("cand_lens", cand_lens)
                if args.no_norm_cand_loss:
                    normed_cand_loss = cand_loss
                else:
                    normed_cand_loss = cand_loss / cand_lens
                # print(normed_cand_loss)
                # exit(0)
                max_res = torch.min(normed_cand_loss, dim=0)
                preds.append(max_res.indices.item())
                gold_loss += normed_cand_loss[cand_label.item()].item()

            gold_loss /= len(losses)
            total_loss += gold_loss

            preds = torch.tensor(preds, dtype=torch.long, device=device)
            gathered_preds = [
                torch.zeros_like(preds)
                for _ in range(mpu.get_data_parallel_world_size())
            ]
            torch.distributed.all_gather(
                gathered_preds, preds.contiguous(), mpu.get_data_parallel_group()
            )
            all_preds.extend(gathered_preds)

            gathered_idx = [
                torch.zeros_like(no_model_batch["idxs"])
                for _ in range(mpu.get_data_parallel_world_size())
            ]
            torch.distributed.all_gather(
                gathered_idx,
                no_model_batch["idxs"].contiguous(),
                mpu.get_data_parallel_group(),
            )
            all_idxs.extend(gathered_idx)

            step += 1

        if step == 0:
            if torch.distributed.get_rank() == 0:
                print(name)
                print(eval_dataset.data_prompts[name][0].name)
                print(len(eval_dataset))

    total_loss /= step

    all_idxs = torch.cat(all_idxs, dim=0).cpu().tolist()
    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    if args.FiD:
        all_scores = torch.cat(all_scores, dim=0)
        print(all_scores.size())
        torch.save(
            all_scores,
            os.path.join(args.save, f"stored_FiD_scores.pt"),
        )

    all_preds_real = []
    all_labels_real = []
    eval_res = {}
    for idxs, pred in zip(all_idxs, all_preds):
        _, _, sid = idxs
        sample = eval_dataset.all_data[name]["data"][sid]
        all_preds_real.append(sample["options"][pred])
        all_labels_real.append(sample["answer"])

    if eval_dataset.data_prompts[name] is None:
        # selfsup
        metric_names = ["Other"]
    else:
        metric_names = eval_dataset.data_prompts[name][0].metadata.metrics
    for metric_name in metric_names:
        if (name, metric_name) in ANSWER_POST_FN:
            all_labels_real, all_preds_real = ANSWER_POST_FN[(name, metric_name)](
                all_labels_real, all_preds_real
            )
        res = T0_METRICS[metric_name](all_labels_real, all_preds_real)
        eval_res.update(res)

    # if save_res:
    #     save_preds_t0(args, name, eval_dataset, train_step, eval_res, all_preds_real, all_labels_real)

    return total_loss, eval_res, all_preds_real, all_labels_real


def load_data(
    args,
    data_prompts,
    split,
    tokenizer,
    ratio=1,
    num=-1,
    few_data_names=None,
    drop_last=True,
):
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if split == "train":
        global_batch_size = args.batch_size * world_size
    elif split == "validation":
        global_batch_size = args.dev_batch_size * world_size
    else:
        global_batch_size = args.eval_batch_size * world_size

    num_workers = args.num_workers

    dataset = T0Dataset(
        args,
        tokenizer,
        data_prompts,
        split,
        ratio=ratio,
        few_data_names=few_data_names,
        num=num,
    )

    if split == "train":
        sampler = RandomSampler(dataset)
        sampler.set_seed(args.seed)
    else:
        sampler = SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(
        sampler=sampler,
        batch_size=global_batch_size,
        drop_last=drop_last,
        rank=rank,
        world_size=world_size,
    )

    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate,
    )

    # Torch dataloader.
    return data_loader, dataset, sampler


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    os.makedirs(args.save, exist_ok=True)

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print("Training Enc-Dec model")
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)

    # Random seeds for reproducability.
    set_random_seed(args.seed)
    device = torch.cuda.current_device()

    # setup tokenizer
    tokenizer = EncDecTokenizer(
        os.path.join(args.tokenizer_path, "spiece.model"), pad_token=args.pad_token
    )

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size

    data_group_names = args.data_names.split("-")
    data_names = []
    for name in data_group_names:
        if name in DATA_GROUP_CONFIG:
            data_names.extend(DATA_GROUP_CONFIG[name])
        else:
            data_names.append(name)

    few_data_names = None
    if args.few_data_names is not None:
        few_data_group_names = args.few_data_names.split("-")
        few_data_names = []
        for name in few_data_group_names:
            if name in DATA_GROUP_CONFIG:
                few_data_names.extend(DATA_GROUP_CONFIG[name])
            else:
                few_data_names.append(name)

    data_prompts = {}
    for name in data_names:
        for ra_name in DATA_RETRIEVAL_AUGMENTATION:
            if ra_name in name:
                DATA_CONFIG[name] = DATA_CONFIG[ra_name]
                DATA_CONFIG[name]["data_dir"] = f"data_hf/{name}/cache"
                break
        if DATA_CONFIG[name].get("selfsup", False):
            data_prompts[name] = None
        else:
            collection = TemplateCollection()
            if "name" in DATA_CONFIG[name]:
                templates = collection.get_dataset(
                    DATA_CONFIG[name]["name"][0], DATA_CONFIG[name]["name"][1]
                )
            else:
                templates = collection.get_dataset(name, None)
            data_prompts[name] = []
            for template_name in templates.all_template_names:
                if (
                    "mmlu" in name or "ai2_arc" in name
                ) and template_name != "heres_a_problem":
                    continue
                if (
                    "marco_qa" in name or "popQA" in name
                ) and template_name != "question_with_instruction":
                    continue
                if (name, template_name) not in DATA_NO_TRAIN:
                    if "marco_qa" in name:
                        prompt = templates[template_name]
                        prompt.metadata.metrics = ["BLEU", "ROUGE"]
                        data_prompts[name].append(prompt)
                    elif "popQA" in name:
                        prompt = templates[template_name]
                        prompt.metadata.metrics = ["popQA"]
                        data_prompts[name].append(prompt)
                    else:
                        data_prompts[name].append(templates[template_name])

    print("All Data group:", data_group_names, "All Data:", data_names)

    if args.do_train:
        train_data_utils = load_data(
            args,
            data_prompts,
            "train",
            tokenizer,
            ratio=args.train_ratio,
            few_data_names=few_data_names,
            num=args.train_num,
        )
        dev_data_utils = {}
        for name in data_prompts:
            if DATA_CONFIG[name].get("selfsup", False):
                dev_data_utils[name] = [
                    load_data(
                        args,
                        {name: None},
                        "validation",
                        tokenizer,
                        ratio=args.dev_ratio,
                        few_data_names=few_data_names,
                        num=args.dev_num,
                    )
                ]
            else:
                if (name, None) not in DATA_NO_VALID:
                    dev_data_utils[name] = []
                    for template in data_prompts[name]:
                        if (name, template.name) not in DATA_NO_VALID:
                            dev_data_utils[name].append(
                                load_data(
                                    args,
                                    {name: [template]},
                                    "validation",
                                    tokenizer,
                                    ratio=args.dev_ratio,
                                    few_data_names=few_data_names,
                                    num=args.dev_num,
                                )
                            )

        if args.train_iters == -1:
            args.train_iters = (
                len(train_data_utils[1])
                * args.epochs
                // (
                    mpu.get_data_parallel_world_size()
                    * args.batch_size
                    * args.gradient_accumulation_steps
                )
            )
        if args.save_interval == -1:
            args.save_interval = len(train_data_utils[1]) // (
                mpu.get_data_parallel_world_size()
                * args.batch_size
                * args.gradient_accumulation_steps
            )
        if args.eval_interval == -1:
            args.eval_interval = len(train_data_utils[1]) // (
                mpu.get_data_parallel_world_size()
                * args.batch_size
                * args.gradient_accumulation_steps
            )
    else:
        args.train_iters = 10  # a magic number

    log_string = "Total train epochs {} | Total train iters {} | ".format(
        args.epochs, args.train_iters
    )
    print_rank_0(log_string)
    save_rank_0(args, log_string)

    # Model, optimizer, and learning rate.
    prompt_config = None
    if args.prompt_tune:
        with open(args.prompt_config, "r") as f:
            prompt_config = json.load(f)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        args,
        tokenizer.vocab_size,
        ds_config,
        set_optim=args.do_train,
        prompt_config=prompt_config,
    )

    if args.do_train:
        train(
            args,
            data_names,
            tokenizer,
            model,
            optimizer,
            lr_scheduler,
            train_data_utils,
            dev_data_utils,
            device,
        )

    if args.do_eval:
        for name in data_names:
            if (name, None) not in DATA_NO_EVAL:
                eval_loss_prompt = []
                eval_res_prompt = []
                eval_preds_prompt = []
                eval_labels_prompt = []
                eval_prompt_names = []
                for template in data_prompts[name]:
                    if (name, template.name) not in DATA_NO_EVAL:
                        eval_data_utils = load_data(
                            args,
                            {name: [template]},
                            "validation",
                            tokenizer,
                            ratio=args.test_ratio,
                            few_data_names=few_data_names,
                            num=args.test_num,
                        )
                        eval_dataloader, eval_dataset, _ = eval_data_utils
                        eval_prompt_names.append(
                            eval_dataset.all_data[name]["prompt_names"][0]
                        )
                        if (
                            eval_dataset.data_prompts[name][0].answer_choices
                            is not None
                            and (name, template.name) not in DATA_EVAL_GEN
                        ):
                            eval_func = evaluate_rank
                        else:
                            eval_func = evaluate_gen
                        eval_loss, eval_res, eval_preds, eval_labels = eval_func(
                            args,
                            tokenizer,
                            name,
                            eval_dataset,
                            eval_dataloader,
                            model,
                            device,
                            mode="test",
                            save_res=True,
                        )
                        eval_loss_prompt.append(eval_loss)
                        eval_res_prompt.append(eval_res)
                        eval_preds_prompt.append(eval_preds)
                        eval_labels_prompt.append(eval_labels)

                avg_eval_res = {
                    k: np.mean([res[k] for res in eval_res_prompt])
                    for k in eval_res_prompt[0]
                }
                log_string = (
                    "Eval result: loss: {:.6} | avg_res: {} | all_res: {}".format(
                        np.mean(eval_loss_prompt), avg_eval_res, eval_res_prompt
                    )
                )
                print_rank_0(log_string)
                save_rank_0(args, log_string)
                save_preds_t0(
                    args,
                    name,
                    eval_prompt_names,
                    0,
                    eval_res_prompt,
                    eval_preds_prompt,
                    eval_labels_prompt,
                )


if __name__ == "__main__":
    main()
