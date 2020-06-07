#!/bin/sh

LOG_ROOT='logs'

python utils/singlemessage_format.py

echo "\n Running all models: baselines, followed to logistic regression, followed by neural. \n Comment out uneeded commands in the script if needed.  \n A GPU is recommended for the neural models. \n \n"

echo "Human Baseline:"
#Human baseline
python diplomacy/models/human_baseline.py

echo "\nRandom and Majority Class Baselines"
#Random and Majority
python diplomacy/models/random_and_majority_baselines.py

#Harbringers
echo "\nHarbringers: Actual Lie"
echo "Without Power"
python diplomacy/models/harbringers.py s n #actual_lie, no power
echo "With Power"
python diplomacy/models/harbringers.py s y #actual_lie, yes power

echo "\nHarbringers: Suspected Lie"
echo "Without Power"
python diplomacy/models/harbringers.py r n #suspected_lie, no power
echo "With Power"
python diplomacy/models/harbringers.py r y #suspected_lie, no power

echo "Bag of Words Log Reg: Actual Lie"
#Bag of words
echo "Without Power"
python diplomacy/models/bagofwords.py s n #suspected_lie, no power
echo "With Power"
python diplomacy/models/bagofwords.py s y #suspected_lie, use power

echo "\nBag of Words Log Reg: Suspected Lie:"
echo "Without Power"
python diplomacy/models/bagofwords.py r n #suspected_lie, no power
echo "With Power"
python diplomacy/models/bagofwords.py r y #suspected_lie, use power


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
