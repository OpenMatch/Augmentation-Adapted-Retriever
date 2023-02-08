# Augmentation-Adapted Retriever (AAR)

## 1 Environment

The code requires the CUDA10.2 toolkit.

##### Install basic dependencies

```bash
pip install -r requirements.txt
```

##### Install apex

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

##### Fix DeepSpeed

Since there exist some **bugs** in DeepSpeed, you need to make some little modifications to this package. Specifically, you need to modify two lines of code in `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/zero/stage1.py` and `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/engine.py`. We provide the modified `tools/ds_fix/stage1.py` and `tools/ds_fix/engine.py` in our repo. You can simply replace `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/zero/stage1.py` with `stage1.py` and `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/engine.py` with `engine.py` that we provided.

## 2 Dataset

We provide the preprocessed data MMLU (target task 1), PopQA (target task 2), and MSMARCO QA (source task) via this [link](TODO).

Please download and unzip it into `data_hf/`.

## 3 Base Models

The original base model is obtained from HuggingFace. Before running the code, please use the transforming scripts to transfer the original pytorch_model.bin model checkpoints to fit in our DeepSpeed + Megatron framework:

```bash
mkdir -p checkpoints/flan-t5-base/t5-MP1

python tools/transform.py \ --hf_path ${PATH_TO_PYTORCH_MODLE_BIN} --save_path "./checkpoints/flan-t5-base/t5-MP1" --half
```

**Note that our base model is the [FLAN-T5 base](https://huggingface.co/google/flan-t5-base).**

## 4 Run the Code

All scripts are in the directory `scripts`.

Before running the code, please first change the `WORKING_DIR` to the current directory of this repo.

If the checkpoint is successfully loaded, the log printed to the stdout should contain messages like `successfully loaded /path-to-checkpoint/t5-MP1/mp_rank_00_model_states.pt`. Otherwise, `WARNING: could not find the metadata file /***/latest_checkpointed_iteration.txt will not load any checkpoints and will start from random` will display. Note that when you successfully load the model, you will see messages like `The following zero checkpoints paths are missing: ['/path-to-checkpoint/200000/zero_pp_rank_0_mp_rank_00_optim_states.pt',...` which mean optimizer states are not loaded. This **DOES NOT** affect the use of model inference and you can just ignore it.

### Zero-shot Evaluation

Running following scripts can reproduce our main results on MMLU and PopQA. If you want to use unassisted versions of LMs, please change the `passage_num` to 0 at first.

For Flan-T5-Base:

```bash
bash scripts/LM/zs_base.sh
```

For Flan-T5-Large:

```bash
bash scripts/LM/zs_large.sh
```

For Flan-T5-XL:

```bash
bash scripts/LM/zs_xl.sh
```

For InstructGPT:

```bash
bash scripts/LM/zs_gpt.sh
```

### Augmentation-Adapted Training

We take T5-ANCE as the retriever backbone in our scripts and you can modify the `model_name_or_path` parameter to specify your own retriever backbone.

First prepare the LM-preferred and human-preferred documents for the augmentation-adapted training:

```bash
bash scripts/Retriever/pre_pipeline.sh
```

Then train the augmentation-adapted retriever (AAR):

```bash
bash scripts/Retriever/train.sh
```

Finally get the relevant documents for target tasks using AAR:

```bash
bash scripts/Retriever/post_pipeline.sh
```
