#!/bin/sh

LOG_ROOT='logs'

python utils/singlemessage_format.py

#Human baseline
python diplomacy/models/human_baseline.py

#Random and Majority
python diplomacy/models/random_and_majority_baselines.py

#Bag of words
python diplomacy/models/bagofwords.py r n #suspected_lie, do not use power

#Harbringers 
python diplomacy/models/harbringers.py s y #actual_lie, use power

#Neural models for actual lie
for name in "lstm" "contextlstm" "contextlstm+power" "bert+context" "bert+context+power"
do
    logdir="${LOG_ROOT}/actual_lie/${name}"
    config="configs/actual_lie/${name}.jsonnet"
    allennlp train -f --include-package diplomacy -s $logdir $config
done

#Neural models for suspected_lie lie
for name in "lstm" "contextlstm" "contextlstm+power" "bert+context" "bert+context+power"
do
    logdir="${LOG_ROOT}/suspected_lie/${name}"
    config="configs/suspected_lie/${name}.jsonnet"
    allennlp train -f --include-package diplomacy -s $logdir $config
done
