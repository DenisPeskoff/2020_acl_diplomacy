{
    "dataset_reader": {
        "type": "diplomacy_reader",
        "label_key": "receiver_labels",
        "lazy": false,
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "do_lowercase": true,
                "max_pieces": 20,
                "pretrained_model": "bert-base-uncased",
                "use_starting_offsets": true
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "do_lowercase": true,
            "model_name": "bert-base-uncased"
        },
        "use_game_scores": false
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
            "input_size": 768
        },
        "dropout": "0.1",
        "embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": [
                    "bert",
                    "bert-offsets"
                ]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "bert-base-uncased"
                }
            }
        },
        "message_encoder": {
            "type": "bert_pooler",
            "pretrained_model": "bert-base-uncased",
            "requires_grad": true
        },
        "pos_weight": "20",
        "use_game_scores": false
    },
    "test_data_path": "data/test.jsonl",
    "train_data_path": "data/train.jsonl",
    "validation_data_path": "data/validation.jsonl",
    "trainer": {
        "cuda_device": 0,
        "grad_clipping": 1,
        "num_epochs": 15,
        "optimizer": {
            "type": "adam",
            "lr": "0.0003"
        },
        "patience": 10,
        "validation_metric": "+macro_fscore"
    },
    "evaluate_on_test": true
}
