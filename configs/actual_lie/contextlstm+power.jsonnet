{
    "pytorch_seed": 1994,
    "numpy_seed": 1994,
    "random_seed": 1994,
    "dataset_reader": {
        "type": "diplomacy_reader",
        "label_key": "sender_labels",
        "lazy": false,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        },
        "use_game_scores": true
    },
    "iterator": {
        "type": "basic",
        "batch_size": 4
    },
    "model": {
        "type": "hierarchical_lstm",
        "conversation_encoder": {
            "type": "lstm",
            "bidirectional": false,
            "hidden_size": 200,
            "input_size": 200
        },
        "dropout": "0.3",
        "embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 200,
                "pretrained_file": "(http://nlp.stanford.edu/data/glove.twitter.27B.zip)#glove.twitter.27B.200d.txt",
                "trainable": false
            }
        },
        "message_encoder": {
            "type": "pooled_rnn",
            "encoder": {
                "type": "lstm",
                "bidirectional": true,
                "hidden_size": 100,
                "input_size": 200
            },
            "poolers": "max"
        },
        "pos_weight": "10",
        "use_game_scores": true
    },
    "train_data_path": "data/train.jsonl",
    "validation_data_path": "data/validation.jsonl",
    "test_data_path": "data/test.jsonl",
    "trainer": {
        "cuda_device": 0,
        "grad_clipping": 1,
        "num_epochs": 15,
        "optimizer": {
            "type": "adam",
            "lr": "0.003"
        },
        "patience": 10,
        "validation_metric": "+macro_fscore"
    },
    "evaluate_on_test": true
}
