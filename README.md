
# LM-Infinite






## Introduction

This is reproduction of the paper
[LM-Infinite: Simple On-the-Fly Length Generalization for Large Language Models](https://arxiv.org/abs/2308.16137)
in PyTorch.
The work is done by [Chi Han](https://glaciohound.github.io), Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, Sinong Wang.

In this paper, the authors propose a simple method, called LM-Infinite, to improve the length generalization of large language models to as long as 128k tokens, without any additional training or parameter updates.
The key idea is to use (1) a $\Lambda$-shaped attention pattern, so that each token only attends to the nearest $L_{pretrain}$ tokens as well as a few starting tokens, and (2) a distance limit $L_{pretrain}$, so that the attention distance is capped at $L_{pretrain}$.
The proposed method is compatible with multiple state-of-the-art language models, including but not limited to LLaMA, Llama-2, GPT-J, MPT-7B series.
LM-Infinite is also computational efficient, with only $O(n)$ time complexity.




## :tada::tada::tada: Now A Drop-in Replacement for HuggingFace Transformers!


We have implemented the LM-Infinite method as a drop-in replacement for HuggingFace Transformers.
After you load the Transformers models, and if it is a Llama model, an MPT model, or a GPT-J model, you can run the following codes to enable LM-Infinite.


For Llama model:
```
from models.llama import convert_llama_model
model = convert_llama_model(model, 4096, 10)
```

For MPT model:
```
from models.mpt_7b import convert_mpt_model
model = convert_mpt_model(model, 4096, 10)
```

For GPT-J model:
```
from models.gpt_j import convert_gpt_j_model
model = convert_gpt_j_model(model, 4096, 10)
```

Then, you can use the model as usual!



## Requirements

- Python 3.11
- PyTorch 2.0.1
- Datasets 2.14.4
- Tokenizers 0.13.3
- Transformers 4.32.1
- SentencePiece 0.1.99
- Evaluate 0.4.0
- Rouge-Score 0.1.2
- Protobuf 3.20.3
- Accelerate 0.22.0
- DeepSpeed 0.10.2
- Tqdm 4.66.1
- Einops 0.6.1

A detailed list of python packages from an Anaconda perspective can be found in `requirements.txt`.
Some packages were installed by `conda` and some by `pip`.
My commands to install the requirements in Anaconda & Pip environment are as follows:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge sentencepiece einops cudatoolkit-dev tqdm ipython datasets evaluate rouge-score protobuf accelerate
pip install transformers deepspeed
```



## Directory Structure

```
├── LICENSE
├── README.md
├── requirements.txt
├── configs
│   └── zero3_efficient_config.json         # config for deepspeed acceleration
├── data
│   ├── generation_metrics.py
│   ├── get_data.py                         # dataset loading and preprocessing
│   ├── passkey_retrieval
│   │   ├── create_passkey_data.py
│   │   ├── create_passkey_data.sh
│   │   └── passkey_retrieval_accuracy.py
│   └── split_pile_file.py                  # split the Pile dataset into task-specific files
├── models
│   ├── constant.py                         # a constant function model
│   ├── get_llama2
│   │   ├── convert_llama_weights_to_hf.py  # convert llama-2 weights to huggingface format
│   │   └── download_llama2.sh
│   ├── get_model.py
│   ├── gpt_j.py
│   ├── lambda_attention.py                 # efficient implementation of lambda attention
│   ├── llama.py
│   ├── model_base.py
│   └── mpt_7b.py
├── scripts
│   ├── combine_evaluate_generation.py
│   ├── combine_results.py
│   ├── eval_downstream_tasks.py            # evaluate on passkey retrieval task
│   ├── eval_generation.py                  # evaluate generation metrics
│   └── eval_ppl_deepspeed.py               # evaluate perplexity
├── utils
│   ├── arguments.py
│   └── utils.py
└── visualization
    ├── plot_nll.py
    ├── position_pca.py
    └── relative_attention_explosion.py
```


## Usage



### Data Preparation


For datasets, you need to prepared a corpus dataset.
If you download the the original Pile source (https://pile.eleuther.ai) to `${PILE_PATH}/test.jsonl.zst` and `${PILE_PATH}/val.jsonl.zst`, run the following commands to extract the compressed dataset.
```
cd ${PILE_PATH}
zstd -d ./ test.jsonl.zst
zstd -d ./ val.jsonl.zst
```
Then run the following commands to split the dataset into task-specific files.
```
cd ${REPOSITORY_ROOT}
mkdir -p ${PILE_PATH}/val
mkdir -p ${PILE_PATH}/test
python data/split_pile_file.py ${PILE_PATH}/val.jsonl ${PILE_PATH}/val
python data/split_pile_file.py ${PILE_PATH}/test.jsonl ${PILE_PATH}/test
```

However the official Pile does not seem to be available for download anymore, so you probably need to figure out another source(e.g., https://huggingface.co/datasets/arxiv_dataset or https://openwebtext2.readthedocs.io/en/latest/).
Alternatively, you can also use your own corpus.
Both two options require you to edit [data/get_data.py](data/get_data.py).







### Model Preparation

For backbone models, the paper uses Llama-2, LLaMA, GPT-J, and MPT-7B.
The last 3 models are directly available on-the-fly from HuggingFace model hub so not action is needed beforehand.
The Llama-2 download key needs to be requested from [Meta AI request form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
Then run the following command
```
bash models/get_llama2/download_llama2.sh
```
and follow prompts to download the checkpoints to `${PATH_TO_LLAMA2_CHECKPOINTS}`.
Then run 
```
python models/get_llama2/convert_llama_weights_to_hf.py \
    --input_dir ${PATH_TO_LLAMA2_CHECKPOINTS} \
    --model_size 7B \
    --output_dir ${PATH_TO_LLAMA2_CHECKPOINTS}/llama-2-7b-hf
```
to convert the llama-2-7b checkpoints to huggingface format.






## Evaluation

The codes requires a `${LOG_DIR}` to store the logs and results.
Please select a directory with enough space.


### Perplexity

Evaluating the perplexity of Llama-2 model on ArXiv test set.

```
TRIAL=llama2-infinite-ArXiv
mkdir -p $LOG_DIR/$TRIAL
CUDA_VISIBLE_DEVICES=0
MASTER_PORT=$(shuf -i 29500-65535 -n 1)
PYTHONPATH=. deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port $MASTER_PORT scripts/eval_ppl_deepspeed.py \
    --deepspeed_config configs/zero3_efficient_config.json \
    --model ${PATH_TO_LLAMA2_CHECKPOINTS}/llama-2-7b-hf --tokenizer_path ${PATH_TO_LLAMA2_CHECKPOINTS} \
    --use_lambda_attention --local_branch 4096 --global_branch 100 --limit_distance 4096 \
    --dataset the_pile --dataset_group ArXiv --split test --dataset_dir ${PILE_PATH} \
    --max_length 32770 \
    --log_dir $LOG_DIR/$TRIAL
```

A brief explanation of the arguments:
- `--model`: the path or name to model. Pass `decapoda-research/llama-7b-hf` to use LLaMA, `mosaicml/mpt-7b` to use MPT-7B, and `EleutherAI/gpt-j-6b` to use GPT-J-6B.
- `--tokenizer_path`: the path to the tokenizer. Remove this argument if not using Llama-2.
- `--use_lambda_attention`: use lambda attention. (Required for LM-Infinite)
- `--local_branch`: the local branch size. 2048 for LLaMA, MPT-7B and GPT-J (Required for LM-Infinite)
- `--global_branch`: the global branch size. Range 10-100 gives generally similar effect. (Required for LM-Infinite)
- `--limit_distance`: the distance limit. 2048 for LLaMA, MPT-7B and GPT-J (Required for LM-Infinite)
- `--dataset`: the dataset name. See [data/get_data.py](data/get_data.py) to figure how to use custom datasets.


If you want to evaluate on vanilla models without LM-Infinite, simply remove the 
`--use_lambda_attention --local_branch 4096 --global_branch 100 --limit_distance 4096 `
argument set.

If you want only to evaluate on a subset of the test set, you can use the `--start_data_from` argument to specify the starting index of the test set, and/or `--max_data_num` to specify the number of examples after that index.


### Generation


Generating evaluation from Llama-2 model on ArXiv test set.

```

TRIAL=llama2-infinite-generate-ArXiv
mkdir -p $LOG_DIR/$TRIAL
CUDA_VISIBLE_DEVICES=0
MASTER_PORT=$(shuf -i 29500-65535 -n 1)
PYTHONPATH=. deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port $MASTER_PORT scripts/eval_generation.py \
    --deepspeed_config configs/zero3_efficient_config.json \
    --model ${PATH_TO_LLAMA2_CHECKPOINTS}/llama-2-7b-hf --tokenizer_path ${PATH_TO_LLAMA2_CHECKPOINTS} \
    --use_lambda_attention --local_branch 4096 --global_branch 100 --limit_distance 4096 \
    --dataset the_pile --dataset_group ArXiv --split test --dataset_dir ${PILE_PATH} \
    --max_length 33000 \
    --max_generation_length 100 --evaluate_metrics --evaluate_positions 4096 8192 12288 16384 \
    --log_dir $LOG_DIR/$TRIAL

```


### Evaluation on Passkey Retrieval Task


First create the dataset with the following command and put to `${PASSKEY_DATA}`.

```
echo $PASSKEY_DATA
mkdir -p ${PASSKEY_DATA}
for MAX_LENGTH in 2048 3072 4096 5120 6144 7168 8192 10240 12288 14335 16384; do
    echo $MAX_LENGTH
    python data/passkey_retrieval/create_passkey_data.py \
        --token-length $MAX_LENGTH \
        --dump-file-path ${PASSKEY_DATA}/${MAX_LENGTH} \
        --tokenizer-path ${PATH_TO_LLAMA2_CHECKPOINTS}
done
```

Evaluating the passkey retrieval task on ArXiv test set with Llama-2 model.

```
MAX_LENGTH=4096
TRIAL=llama2-infinite-passkey-$MAX_LENGTH
mkdir -p $LOG_DIR/$TRIAL
CUDA_VISIBLE_DEVICES=0
MASTER_PORT=$(shuf -i 29500-65535 -n 1)
PYTHONPATH=. deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port $MASTER_PORT scripts/eval_downstream_tasks.py \
    --deepspeed_config configs/zero3_efficient_config.json \
    --model ${PATH_TO_LLAMA2_CHECKPOINTS}/llama-2-7b-hf --tokenizer_path ${PATH_TO_LLAMA2_CHECKPOINTS} \
    --use_lambda_attention --local_branch 4096 --global_branch 100 --limit_distance 4096 \
    --dataset passkey_retrieval --dataset_dir ${PASSKEY_DATA} --dataset_group ${MAX_LENGTH} \
    --max_generation_length 10 --evaluate_metrics \
    --log_dir $LOG_DIR/$TRIAL
```



## Citation

```
@article{han2023lminfinite,
  title={LM-Infinite: Simple On-the-Fly Length Generalization for Large Language Models},
  author={Han, Chi and Wang, Qifan and Xiong, Wenhan and Chen, Yu and Ji, Heng and Wang, Sinong},
  journal={arXiv preprint arXiv:2308.16137},
  year={2023}
}
```
