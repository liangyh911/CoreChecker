# CoreChecker

## Installation

### From Source

#### Our Environment

- Apptainer 1.4.1

- CUDA Toolkit 12.9

- GCC 13.3.0

- 4 $\times$ NVIDIA H100-SXM5 GPUs with 80 GB HBM.

#### Build and launch Apptainer sandbox image

```shell
apptainer build --sandbox nv_megatron_env docker://nvcr.io/nvidia/pytorch:25.04-py3
apptainer shell --nv --writable nv_megatron_env
```

#### Download Source Code

```shell
cd /workspace
git clone https://github.com/liangyh911/CoreChecker
```

#### Download Required Dependencies

```shell
cd ./CoreChecker
unset PIP_CONSTRAINT
pip install -r requirements.txt
```

#### Move the HuggingFace Modeling Scripts from HF-modeling to Huggingface Transformers Package

```shell
cp ./HF-modeling/trainer.py      ./usr/local/lib/python3.12/dist-packages/transformers/trainerpy
cp ./HF-modeling/modeling_llama.py      ./usr/local/lib/python3.12/dist-packages/transformers/models/llama/modeling_llama.py
cp ./HF-modeling/modeling_gemma3.py      ./usr/local/lib/python3.12/dist-packages/transformers/models/gemma3/modeling_gemma3.py
cp ./HF-modeling/modeling_qwen3.py   ./usr/local/lib/python3.12/dist-packages/transformers/models/qwen3/modeling_qwen3.py
cp ./HF-modeling/modeling_gemma2.py   ./usr/local/lib/python3.12/dist-packages/transformers/models/gemma2/modeling_gemma2.py
```

#### Rebuild PyTorch from Source
```shell
pip uninstall torch
alias which=’command -v’
cd ./CoreChecker/pytorch
conda install cmake ninja
pip install mkl-static mkl-include
pip install -r requirements.txt
python setup.py develop
```

#### Rebuild Transformer Engine
```shell
pip uninstall transformer-engine
$ git clone --branch v2.12 https://github.com/NVIDIA/TransformerEngine.git
cd ./TransformerEngine
export NVTE_FRAMEWORK=pytorch
pip install . --no-deps --no-build-isolation
```

#### Reinstall Flash-Attention
```shell
pip uninstall flash-attn
pip install flash-attn==2.7.3 --no-build-isolation --no-deps
```

## Megatron Models and Datasets Preparation

Multi-GPU training requires serval preparation, including downloading the model checkpoint and training datasest, and data preprocessing. The prepartion scripts are provided in ```./megatron_preparation_scripts``` folder. 

You can run following scripts to complete the preparations.

```shell
# Qwen3-8B
bash ./megatron_preparation_scripts/pre_qwen3.sh
# Qwen2.5-7B
bash ./megatron_preparation_scripts/pre_qwen2_5.sh
# Llama3.1-8B
bash ./megatron_preparation_scripts/pre_llama3_1.sh
```

## Usage

### Single-GPU Training Performance Evaluation

To evaluate the runtime overhead and detection accuracy of CoreChecker for single-GPU training, you need to use the scripts in ```./HF_eval_scripts``` folder.

Before running the scripts, make sure to use one GPU if you have multi-GPUs on your device. You can use ```export``` command.

```shell
export CUDA_VISIBLE_DEVICES=0
```

Then you can run the evaluation scripts. Each script performs CoreChecker runtime evluation and faulty SM detection evaluation.

```shell
# Evaluations in Single-GPU Training.
cd ./CoreChecker

export CUDA_VISIBLE_DEVICES=0
# Llama3.2-1B
bash /HF_eval_scripts/eval_llama3_2.sh
# Gemma3-1B
bash /HF_eval_scripts/eval_gemma3.sh
# Qwen3-1.7B
bash /HF_eval_scripts/eval_qwen3.sh
# Gemma2-2B
bash /HF_eval_scripts/eval_gemma2.sh
```

### Multi-GPU Training Performance Evaluation

To evaluate the runtime overhead and detection accuracy of CoreChecker in Multi-GPU training, you need to use the scripts in ```./megatron_eval_scripts``` folder.

The parallelism configuration is Tensor Parallelism (TP) = 2 and Pipeline Parallelism (PP) = 2. Each script performs CoreChecker runtime evluation and faulty SM detection evaluation.

```shell
# Evaluations in Multi-GPU Trainnig.
cd ./CoreChecker

export CUDA_VISIBLE_DEVICES=0,1,2,3
# Llama3.1-8B
$ bash /megatron_eval_scripts/eval_llama3_1.sh
# Qwen2.5-7B
$ bash /megatron_eval_scripts/eval_qwen2_5.sh
# Qwen3-8B
$ bash /megatron_eval_scripts/eval_qwen3.sh
```