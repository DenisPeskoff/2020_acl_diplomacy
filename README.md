# 2020_acl_diplomacy
Repository for ACL 2020 Paper: "It Takes Two to Lie: One to Lie, and One to Listen"

## Setup

```
conda create --name diplomacy python=3.7
conda activate diplomacy
pip install -e .
```

The neural models in this repository are implemented using allennlp. To run a model just change `logdir` and the config file in the command below. All our config files are under `configs/`. 

```
allennlp train -f --include-package diplomacy -s logdir configs/actual_lie/bert+context.jsonnet
```

To run the lstm model (`lstm.jsonnet`), please make sure to run `python utils/singlemessage_format.py` first.

UPDATE: We include full game (without messages) data for the 12 games that is not used in our paper under utils/ExtraGameData.zip


## Citation
```
@inproceedings{Peskov:Cheng:Elgohary:Barrow:Danescu-Niculescu-Mizil:Boyd-Graber-2020,
	Title = {It Takes Two to Lie: One to Lie and One to Listen},
	Author = {Denis Peskov and Benny Cheng and Ahmed Elgohary and Joe Barrow and Cristian Danescu-Niculescu-Mizil and Jordan Boyd-Graber},
	Booktitle = {Association for Computational Linguistics},
	Year = {2020},
	Location = {The Cyberverse Simulacrum of Seattle},
}

## License 
```
This work is licensed under a Creative Commons Attribution 4.0 International License.
