# Final Project

## work division
expriments and reports :
-Sun Shiyuan 1900017802 complete "LLM-pruner" part
-Zhao Chengyang 1900017796 complete "Qlora" part
-He Wenyang 2000013053 complete "GPTQ" part
-Wang Zanming 2100013135 complete "Quip" part

## llm-pruner 

### Environment Setup
First, install torch>=1.7.1 with cuda>=11.1
Then install the requirements via pip install -r llm-pruner/requirements.txt.

### Introduction
  
To the best of our knowledge, LLM-Pruner is the first framework designed for structured pruning of LLMs. it conclude the advantages of theLLMPruner as (i) Task-agnostic compression, where the compressed language model retains its ability to serve as a multi-task solver. (ii) Reduced demand for the original training corpus, where only 50k publicly available samples are needed for compression, significantly reducing the budget for acquiring the training data (iii) Quick compression, where the compression process ends up in three hours. (iv) An automatic structural pruning framework, where all the dependent structures are grouped without the need for any manual design. The experimental results demonstrate that even with the removal of 20% of the parameters, the pruned model maintains 94.97

#### Why LLM-Pruner
- [x] **Task-agnostic compression**: The compressed LLM should retain its original ability as a multi-task solver. 
- [x] **Less training corpus**: In this work, we use only 50k publicly available samples (alpaca) to post-train the LLM.  
- [x] **Efficient compression**: 3 minutes for pruning and 3 hours for post-training.
- [x] **Automatic structural pruning**: Pruning new LLMs with minimal human effort (In progress).

### Quick Start

#### Installation
```
pip install -r requirement.txt

```    

#### 0.Quick Start
```
bash llama_prune.sh

```    

#### 1. Pruning (Discovery Stage + Estimation Stage)
    
:llama: **LLaMA/Llama-2 pruning with ~20% parameters pruned:**
```
python hf_prune.py --pruning_ratio 0.25 \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
      --block_attention_layer_start 4 --block_attention_layer_end 30 \
      --pruner_type taylor \
      --test_after_train \
      --device cpu  --eval_device cuda \
      --save_ckpt_log_name llama_prune 
```
Arguments:
- ``Base model``: Choose the base model from LLaMA or Llama-2 and pass the `pretrained_model_name_or_path` to `--base_model`. The model name is used for `AutoModel.from_pretrained` to load the pre-trained LLM. For example, if you want to use the llama-2 with 13 billion parameters, than pass `meta-llama/Llama-2-13b-hf` to `--base_model`.
- ``Pruning Strategy``: Choose between block-wise, channel-wise, or layer-wise pruning using the respective command options: {--block_wise}, {--channel_wise}, {--layer_wise --layer NUMBER_OF_LAYERS}. For block-wise pruning, specify the start and end layers to be pruned. Channel-wise pruning does not require extra arguments. For layer pruning, use --layer NUMBER_OF_LAYERS to specify the desired number of layers to be kept after pruning.
- ``Importance Criterion``: Select from l1, l2, random, or taylor using the --pruner_type argument. For the taylor pruner, choose one of the following options: vectorize, param_second, param_first, param_mix. By default, param_mix is used, which combines approximated second-order hessian and first-order gradient. If using l1, l2, or random, no extra arguments are required.
- ``Pruning Ratio``: Specifies the pruning ratio of groups. It differs from the pruning rate of parameters, as groups are removed as the minimal units.
- ``Device`` and ``Eval_device``: Pruning and evaluation can be performed on different devices. Taylor-based methods require backward computation during pruning, which may require significant GPU RAM. Our implementation uses the CPU for importance estimation (also support GPU, simply use --device cuda). eval_device is used to test the pruned model.

#### 2. Post-Training (Recover Stage)

* Train using Alpaca with 50,000 samples. Here's an example of training on a single GPU:
```
CUDA_VISIBLE_DEVICES=X python post_training.py --prune_model prune_log/PATH_TO_PRUNE_MODEL/pytorch_model.bin \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \ 
      --learning_rate 1e-4 \ 
      --batch_size 64 \
      --output_dir tune_log/PATH_TO_SAVE_TUNE_MODEL \ 
      --wandb_project llama_tune
```

#### 3. Generation

For the pruned model, simply use the following command to load your model. 
``` 
  pruned_dict = torch.load(YOUR_CHECKPOINT_PATH, map_location='cpu')
  tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
```

## Quip

### Environment Setup
First, install torch>=2.1.1 with cuda>=12.1
Then install the requirements via pip install -r quip-sharp/requirements.txt.

### How to Run
download Llama-2-7b model
download  RedPajama-Data-1T-Sample and wikitext-2-raw-v1 for evaluation
then run `bash quip-sharp/experiment.sh`

### Open-source Code URL
The code is from the official QuIP\# open-source code base.
Link:
- https://github.com/Cornell-RelaxML/quip-sharp


# Environment Setup

Please install the following dependencies to run the code for reproduction. You need to run 

- torch==2.1.0+cu118
- bitsandbytes==0.40.0
- transformers==4.31.0
- peft==0.4.0
- accelerate==0.21.0
- einops==0.6.1
- evaluate==0.4.0
- scikit-learn==1.2.2
- sentencepiece==0.1.99
- wandb==0.15.3

You can also directly use `pip install -v -r requirements.txt` to setup the environment.

## Qlora

### Environment Setup

### Finetuning

```bash
bash finetune_guanaco_llama_2_7b.sh
```

This code fine-tune official LLaMA-2-7B model from Meta on the OASST1 dataset using the QLoRA technique. To access to the official LLaMA-2-7B model, you need to add your own valid huggingface API key to the in `HF_TOKEN` in the `finetune_guanaco_llama_2_7b.sh` file.

### Inferencing

```bash
bash generate.sh
```

This code generates the inference results on the Alpaca dataset. To access to the official LLaMA-2-7B model, you need to add your own valid huggingface API key to the in `HF_TOKEN` in the `generate.sh` file.


### Open-source Code URL

The code is from the official QLoRA open-source code base.

Link:

- https://github.com/artidoro/qlora

## gptq-pq

The main author of this section is He Wenyang, focusing on the application of Product Quantization (PQ) technology to the quantization of large language models. This technique is commonly used in recommendation systems, advertising, and search scenarios to compress embeddings and quickly calculate the distance between two embeddings. However, its application in large language models is still limited. *This idea was proposed by Professor Tong Yang and is expected to be my graduation project under his guidance.*

### Introduction

Large language models have tens of billions of parameters. To achieve efficient inference, the entire model needs to be present in GPU memory. Therefore, one approach is to compress the model's parameters, representing a parameter with fewer bits. This allows storing larger models in less GPU memory and also improves inference efficiency by saving memory bandwidth.

The simplest and most intuitive quantization method is converting floating-point numbers into fixed-point numbers with certain precision; see https://arxiv.org/abs/2208.07339. Rounding parameter precision directly to 8 bits has little impact on the model's effectiveness. However, as the quantization bit number decreases, its impact on the model increases, and performance starts to degrade significantly when it reaches 4 bits. To quantize models at 4 bits or even lower precision, more complex quantization techniques are needed.

Techniques like GPTQ (https://github.com/IST-DASLab/gptq) are set as "Post-Training Quantization," completing model quantization after training without further training. The development history of this technique can be found in papers, being an efficiency optimization based on the Optimal Brain Compression work. The core ideas of this kind of work can be summarized as:

1. The goal is to minimize the error between pre-quantization and post-quantization output values of each network layer.

2. The Hessian of the input vector is uneven, and the Hessian determines which directions of error in each layer's weight matrix will significantly affect the layer's output.
3. Each row of the weight matrix can be viewed as an independent quantization problem; there is no Hessian between rows of the weight matrix, and the Hessian between different rows is the same.

Based on these ideas, OBC's approach is to greedily quantize the weight with the least impact each time and adjust all other unquantized weights along the Hessian direction. GPTQ, building on this, notes that the order of quantization is not important, and it can quantize an entire column at once instead of a single weight, significantly improving efficiency. It includes most tricks required for model quantization, making its code a good foundation for Product Quantization.

## Algorithm

The PQ algorithm requires three hyperparameters: dimension $d$, number of centroids  $n$, and group size $g$.  During quantization, the algorithm clusters every $g$ $d$-dimensional vectors into $n$ categories.  After quantization, $\log_2(n)$ bits are needed to store which centroid each group of $g$ vectors belongs to; additionally, $16dn$ bits are required to store the centroids themselves. The total is $g\log_2(n)+16dn$ bits for $gn$ weights, for $\frac{g\log_2(n)+16dn}{gd}$ bits per quantized weight.

The current parameters used are $(d,n,g)=(2,64,1024)$, corresponding to 4-bit quantization; each two-dimensional vector is quantized into 6 bits, while every 1024 vectors require an additional storage space for 64 half-precision vectors.

For the weight matrix $W\in M_{n,d}$, we follow GPTQ's approach of quantizing by column, but not one column at a time. Instead, we quantize $d$ columns, applying the PQ algorithm to these $d$ columns.

### Experiment

Due to hardware resource limitations, I conducted two sets of quantization experiments, both with the llama2-7b model. The first group used the original GPTQ 3-bit algorithm, and the second group applied the PQ algorithm on top of GPTQ, where $(d,n,g)=(2,64,1024)$. During quantization, the L2 error of post-quantization weight changes and the L2 error under a set of random inputs were calculated. The results are as follows:

![](fig.png)

Another metric tested is the same model's perplexity on WikiText-2. The PQ quantized model's perplexity was 5.7856, while the GPTQ quantized model's perplexity was 7.6006。

The PQ algorithm outperforms the GPTQ algorithm in terms of both error in each layer and final perplexity, with a fourfold smaller squared error. It should be noted that this comparison is not fair, as the average quantization bit number of the PQ algorithm is 4bit, while that of the GPTQ algorithm is 3bit. However, the PQ algorithm still has significant room for improvement: the extra 1bit comes from recording 64 centroids for every 1024 parameters. On the one hand, these centroids could be further quantized to 8-bit, which existing research proves would hardly affect model accuracy; on the other hand, removing the centroids and using predefined ones could save this storage space, which will be the focus of future research.

### Conclusion

The PQ algorithm shows strong potential, but the recently released new algorithm [QuIP#](https://cornell-relaxml.github.io/quip-sharp/) adopts a very similar approach, using geometrically superior predefined codebooks, quantizing 8 dimensions at a time, comparable to a PQ algorithm with $(d,n,g)=(8,65536,\infty)$,  and its ideas might be more worth learning from.

## Checkpoint Download Link and File Description

`result-88.ipynb` is the code and output used for quantization with the PQ algorithm, modifying the GPTQ code with `gptq-with-pq2-64-1024.patch`. `result-89.ipynb` is the code and output used for 3-bit GPTQ quantization, modifying the GPTQ code with `gptq.patch` (only fixing bugs in the code). Both notebooks can be run automatically on machines with good network conditions (machines can be rented from vast.ai).

Since the algorithm in `result-89.ipynb` is consistent with the official GPTQ, it can be reproduced directly from the official code, so no checkpoint is provided here. The model weights quantized using the PQ algorithm are as follows (weights are still stored in fp16 format and can be directly loaded with huggingface transformer):

https://f004.backblazeb2.com/file/share-rxwfytxfqp/RdcbHz4f3gaiVMte/2024-01-20-llama-quantized.tar
