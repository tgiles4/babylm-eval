import math

import torch
import numpy as np

from evaluation_pipeline.utils import get_logits
from evaluation_pipeline.sentence_zero_shot.diffusion_likelihood import get_log_likelihood

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_p(sentence, word, model, tokenizer):  # gets p of word (word) given context. Relies on model and tokenizer.
    inpts = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inpts)
        logits = get_logits(outputs)[:, -1, :].cpu()
    target_id = tokenizer(word, add_special_tokens=False)["input_ids"][0]
    p = torch.softmax(logits[0], dim=-1)[target_id].item()
    return p


def get_p_mntp(sentence, word, model, tokenizer, num_mask_tokens=3):  # gets p of word (word) given context. Relies on model and tokenizer.
    inpts = tokenizer("".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])]), return_tensors="pt").to(DEVICE)
    position_of_pred = -(num_mask_tokens + 1) if inpts.input_ids[:, -1] == tokenizer.mask_token_id else -(num_mask_tokens + 2)
    with torch.no_grad():
        outputs = model(**inpts)
        logits = get_logits(outputs)[:, position_of_pred, :].cpu()
    target_id = tokenizer(word, add_special_tokens=False)["input_ids"][0]
    p = torch.softmax(logits[0], dim=-1)[target_id].item()
    return p


def get_p_mlm(sentence, word, model, tokenizer, num_mask_tokens=3):  # gets p of word (word) given context. Relies on model and tokenizer.
    inpts = tokenizer("".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])]), return_tensors="pt").to(DEVICE)
    position_of_pred = -num_mask_tokens if inpts.input_ids[:, -1] == tokenizer.mask_token_id else -(num_mask_tokens + 1)
    with torch.no_grad():
        outputs = model(**inpts)
        logits = get_logits(outputs)[:, position_of_pred, :].cpu()
    target_id = tokenizer(word, add_special_tokens=False)["input_ids"][0]
    p = torch.softmax(logits[0], dim=-1)[target_id].item()
    return p


def get_p_enc_dec(sentence, word, model, tokenizer):  # gets p of word (word) given context. Relies on model and tokenizer.
    if tokenizer.cls_token is not None:
        cls_token = [tokenizer.cls_token_id]
        att_cls = [1]
    else:
        cls_token = []
        att_cls = []
    if tokenizer.mask_token is not None:
        mask_token = tokenizer.mask_token_id
    else:
        if tokenizer.additional_special_tokens is not None:
            mask_token = tokenizer.additional_special_tokens_id[0]
        else:
            raise "Unknown mask token, please specify it in the tokenizer!"
    if tokenizer.bos_token is not None:
        bos_token = tokenizer.bos_token
    else:
        bos_token = tokenizer.additional_special_tokens[0]
    if tokenizer.eos_token_id is not None:
        eos_token = [tokenizer.eos_token_id]
        att_append = [1]
    else:
        eos_token = []
        att_append = []
    inpts = tokenizer(sentence, add_special_tokens=False)
    input_ids = cls_token + inpts["input_ids"] + [mask_token] + eos_token
    input_ids = torch.LongTensor(input_ids)[None, :]
    attention_mask = att_cls + inpts["attention_mask"] + [1] + att_append
    attention_mask = torch.LongTensor(attention_mask)[None, :]
    dec_inpts = tokenizer(bos_token, add_special_tokens=False, return_tensors="pt")
    dec_input_ids = dec_inpts["input_ids"]
    dec_attention_mask = dec_inpts["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE), decoder_input_ids=dec_input_ids.to(DEVICE), decoder_attention_mask=dec_attention_mask.to(DEVICE))
        logits = get_logits(outputs)[:, 0, :].cpu()
    target_id = tokenizer(word, add_special_tokens=False)["input_ids"][0]
    p = torch.softmax(logits[0], dim=-1)[target_id].item()
    return p


def get_p2(sentence, word, model, tokenizer):  # as get_p if len(tokenizer(word)) == 1; else, sums logP of subword tokens
    inpts = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inpts)
        logits = get_logits(outputs)[:, -1, :].cpu()
    target = tokenizer(word, add_special_tokens=False)["input_ids"]  # Check whether tokenizer adds a whitespace to the beginning of input.
    if len(target) == 1:
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        return p, 0
    else:
        out_p = []
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        out_p.append(p)
        sentence = sentence + tokenizer.decode(target_id)
        for token in target[1:]:
            t = tokenizer.decode(token)
            p = get_p(sentence, t, model, tokenizer)
            out_p.append(p)
            # print(sentence, "--"+t, p)
            sentence = sentence + t
        p_multi = np.prod(out_p)
        return p_multi, 1


def get_p2_mlm(sentence, word, model, tokenizer, num_mask_tokens=3):  # as get_p if len(tokenizer(word)) == 1; else, sums logP of subword tokens
    inpts = tokenizer("".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])]), return_tensors="pt").to(DEVICE)
    position_of_pred = -num_mask_tokens if inpts.input_ids[:, -1] == tokenizer.mask_token_id else -(num_mask_tokens + 1)
    with torch.no_grad():
        outputs = model(**inpts)
        logits = get_logits(outputs)[:, position_of_pred, :].cpu()
    target = tokenizer(word, add_special_tokens=False)["input_ids"]  # Check whether tokenizer adds a whitespace to the beginning of input.
    if len(target) == 1:
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        return p, 0
    else:
        out_p = []
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        out_p.append(p)
        sentence = sentence + tokenizer.decode(target_id)
        for token in target[1:]:
            t = tokenizer.decode(token)
            p = get_p_mlm(sentence, t, model, tokenizer, num_mask_tokens)
            out_p.append(p)
            # print(sentence, "--"+t, p)
            sentence = sentence + t
        p_multi = np.prod(out_p)
        return p_multi, 1


def get_p2_mntp(sentence, word, model, tokenizer, num_mask_tokens=3):  # as get_p if len(tokenizer(word)) == 1; else, sums logP of subword tokens
    inpts = tokenizer("".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])]), return_tensors="pt").to(DEVICE)
    position_of_pred = -(num_mask_tokens + 1) if inpts.input_ids[:, -1] == tokenizer.mask_token_id else -(num_mask_tokens + 2)
    with torch.no_grad():
        outputs = model(**inpts)
        logits = get_logits(outputs)[:, position_of_pred, :].cpu()
    target = tokenizer(word, add_special_tokens=False)["input_ids"]  # Check whether tokenizer adds a whitespace to the beginning of input.
    if len(target) == 1:
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        return p, 0
    else:
        out_p = []
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        out_p.append(p)
        sentence = sentence + tokenizer.decode(target_id)
        for token in target[1:]:
            t = tokenizer.decode(token)
            p = get_p_mntp(sentence, t, model, tokenizer, num_mask_tokens)
            out_p.append(p)
            # print(sentence, "--"+t, p)
            sentence = sentence + t
        p_multi = np.prod(out_p)
        return p_multi, 1


def get_p2_diffusion(sentence, word, model, tokenizer, mc_num=128, mc_batch_size=16):
    """P(word | sentence) via LLaDA Eq. 6 conditional log-likelihood.

    Prompt = tokenized context; answer = tokenized word (no specials). Returns
    exp(log-likelihood) so callers can use -log(p) as surprisal, matching MLM/causal.
    Single-token answers use mc_num=1 (LLaDA); multi-token uses the full budget.
    """
    prompt_ids = tokenizer(sentence, add_special_tokens=True)["input_ids"]
    word_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    if not word_ids:
        raise ValueError(f"Empty tokenization for word={word!r}")

    seq = torch.tensor(prompt_ids + word_ids, dtype=torch.long, device=DEVICE)
    prompt_index = torch.zeros(seq.shape[0], dtype=torch.bool, device=DEVICE)
    prompt_index[: len(prompt_ids)] = True

    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Diffusion reading requires a mask_token_id on model or tokenizer.")

    effective_mc = 1 if len(word_ids) == 1 else mc_num
    effective_bs = 1 if effective_mc == 1 else min(mc_batch_size, effective_mc)
    if effective_mc % effective_bs != 0:
        effective_bs = 1

    log_lik = get_log_likelihood(
        model,
        seq,
        prompt_index,
        mask_id=mask_id,
        mc_num=effective_mc,
        mc_batch_size=effective_bs,
    )
    p = math.exp(log_lik)
    return max(p, 1e-300), int(len(word_ids) > 1)


def get_p2_enc_dec(sentence, word, model, tokenizer):  # as get_p if len(tokenizer(word)) == 1; else, sums logP of subword tokens
    if tokenizer.cls_token is not None:
        cls_token = [tokenizer.cls_token_id]
        att_cls = [1]
    else:
        cls_token = []
        att_cls = []
    if tokenizer.mask_token is not None:
        mask_token = tokenizer.mask_token_id
    else:
        if tokenizer.additional_special_tokens is not None:
            mask_token = tokenizer.additional_special_tokens_id[0]
        else:
            raise "Unknown mask token, please specify it in the tokenizer!"
    if tokenizer.bos_token is not None:
        bos_token = tokenizer.bos_token
    else:
        bos_token = tokenizer.additional_special_tokens[0]
    if tokenizer.eos_token_id is not None:
        eos_token = [tokenizer.eos_token_id]
        att_append = [1]
    else:
        eos_token = []
        att_append = []
    inpts = tokenizer(sentence, add_special_tokens=False)
    input_ids = cls_token + inpts["input_ids"] + [mask_token] + eos_token
    input_ids = torch.LongTensor(input_ids)[None, :]
    attention_mask = att_cls + inpts["attention_mask"] + [1] + att_append
    attention_mask = torch.LongTensor(attention_mask)[None, :]
    dec_inpts = tokenizer(bos_token, add_special_tokens=False, return_tensors="pt")
    dec_input_ids = dec_inpts["input_ids"]
    dec_attention_mask = dec_inpts["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE), decoder_input_ids=dec_input_ids.to(DEVICE), decoder_attention_mask=dec_attention_mask.to(DEVICE))
        logits = get_logits(outputs)[:, 0, :].cpu()
    target = tokenizer(word, add_special_tokens=False)["input_ids"]  # Check whether tokenizer adds a whitespace to the beginning of input.
    if len(target) == 1:
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        return p, 0
    else:
        out_p = []
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        out_p.append(p)
        sentence = sentence + tokenizer.decode(target_id)
        for token in target[1:]:
            t = tokenizer.decode(token)
            p = get_p_enc_dec(sentence, t, model, tokenizer)
            out_p.append(p)
            # print(sentence, "--"+t, p)
            sentence = sentence + t
        p_multi = np.prod(out_p)
        return p_multi, 1
