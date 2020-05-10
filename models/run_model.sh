#!/bin/sh

LOGDIR=logs/actual_lie/lstm
CONFIG=configs/actual_lie/lstm.jsonnet

mkdir -p $LOGDIR
allennlp train -f --include-package diplomacy -s $LOGDIR $CONFIG
