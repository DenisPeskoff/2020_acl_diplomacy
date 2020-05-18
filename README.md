# 2020_acl_diplomacy
Repository for ACL 2020 Paper: "It Takes Two to Lie: One to Lie, and One to Listen"

The neural models in this repository are implemented using allennlp v0.9.0. To run a model just change `logdir` and the config file in the command below. All our config files are under `configs/`. 

```
allennlp train -f --include-package diplomacy -s logdir configs/actual_lie/bert+context.jsonnet
```

To run the lstm model (`lstm.jsonnet`), please make sure to run `python singlemessage_format.py` first.

## Citation
```
@inproceedings{Peskov:Cheng:Elgohary:Barrow:Danescu-Niculescu-Mizil:Boyd-Graber-2020,
	Title = {It Takes Two to Lie: One to Lie and One to Listen},
	Author = {Denis Peskov and Benny Cheng and Ahmed Elgohary and Joe Barrow and Cristian Danescu-Niculescu-Mizil and Jordan Boyd-Graber},
	Booktitle = {Association for Computational Linguistics},
	Year = {2020},
	Location = {The Cyberverse Simulacrum of Seattle},
}
```
