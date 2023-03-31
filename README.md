# T5-Zero

## Introduction
This repository uses the T5 model architecture for various NLP tasks.

## Installation
Please install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (miniconda recommended) for the environment manager.<br>

Once conda is installed, you can create and activate the environment using the following commads.

```
conda env create -f t5-zero.yml
conda activate t5-zero
```

## Usage
Choose you config file from the config folder.<br>
Then modify the train.sh script. Then simply run

```
bash train.sh
```