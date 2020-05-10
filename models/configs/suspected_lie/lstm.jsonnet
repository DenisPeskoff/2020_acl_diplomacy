{
    "dataset_reader": {
        "lazy": false,
        "sender_annotation": false,
        "token_indexers": {
            "tokens": {
                "lowercase_tokens": true,
                "namespace": "tokens",
                "type": "single_id"
            }
        },
        "type": "message_reader"
    },
    "evaluate_on_test": true,
    "iterator": {
        "batch_size": 32,
        "sorting_keys": [
            [
                "message",
                "num_tokens"
            ]
        ],
        "type": "bucket"
    },
    "model": {
        "dropout": 0.5,
        "embedder": {
            "tokens": {
                "embedding_dim": 300,
                "pretrained_file": "(http://nlp.stanford.edu/data/glove.840B.300d.zip)#glove.840B.300d.txt",
                "trainable": true,
                "type": "embedding"
            }
        },
        "encoder": {
            "encoder": {
            "bidirectional": true,
            "hidden_size": 100,
            "input_size": 300,
            "type": "lstm"
            },
            "type": 'pooled_rnn',
            "poolers": "max",
        },
        posclass_weight: 15.0,
        use_power: false,
        "type": "lie_detector"
    },
    "test_data_path": '../data/test_sm.jsonl',
    "train_data_path": '../data/train_sm.jsonl',
    "validation_data_path": '../data/validation_sm.jsonl',
    "trainer": {
        "cuda_device": -1,
        "grad_clipping": 1.0,
        "num_epochs": 15,
        "patience": 5,
        "optimizer": {
            "lr": 0.003,
            "type": "adam"
        },
        "validation_metric": "+macro_fscore"
    }
}