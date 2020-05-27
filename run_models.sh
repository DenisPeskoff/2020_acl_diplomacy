#!/bin/sh

LOG_ROOT='logs'

python utils/singlemessage_format.py

#Human baseline
python models/models/human_baseline.py

#Random and Majority
python models/models/random_and_majority_baselines.py

#Bag of words
python models/models/bagofwords.py r n #suspected_lie, do not use power

#Harbringers 
python models/models/harbringers.py s y #actual_lie, use power

#Neural models for actual lie
for name in "lstm" "contextlstm" "contextlstm+power" "bert+context" "bert+context+power"
do
    logdir="${LOG_ROOT}/actual_lie/${name}"
    config="models/configs/actual_lie/${name}.jsonnet"
    allennlp train -f --include-package models -s $logdir $config
done

#Neural models for suspected_lie lie
for name in "lstm" "contextlstm" "contextlstm+power" "bert+context" "bert+context+power"
do
    logdir="${LOG_ROOT}/suspected_lie/${name}"
    config="models/configs/suspected_lie/${name}.jsonnet"
    allennlp train -f --include-package models -s $logdir $config
done
