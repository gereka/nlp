local bert_model = "/home/pra.agerek/shared/uk1577/models/bert_base_turkish_uncased/";

{
    "pytorch_seed": 4303,
    "dataset_reader": {
        "type": "mip_dataset_reader2",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model
        },
        "label_index": -1,
        //"lazy": true,
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model
            }
        }
    },
  "train_data_path": "datasets/ekim49k_UKK_train.txt",
  "validation_data_path": "datasets/ekim49k_UKK_dev.txt",
  "test_data_path": "datasets/ekim49k_UKK_test.txt",
    "model": {
        "namespace": 'tags',
        "type": "basic_classifier",
        "text_field_embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
        "seq2vec_encoder": {
           "type": "bert_pooler",
           "pretrained_model": bert_model,
           "requires_grad": true,
       "dropout": 0.1,
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["tokens"],
            "batch_size": 16,
            //"max_instances_in_memory": 10000
        }
    },
    "trainer": {
        "num_epochs": 40,
        "patience": 10,
        "validation_metric": "+accuracy",
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_steps_per_epoch": 3088,
            "cut_frac": 0.06
        },
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-5,
            "weight_decay": 0.2,
        },
        "cuda_device": 0
    },
  "evaluate_on_test": true
}
