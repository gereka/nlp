local bert_model = "/home/gereka/data/models/convbert-base-turkish-cased/";

{
    "pytorch_seed": 4303,
    "dataset_reader": {
        "type": "csvreader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
	    "max_length": 510 //accounting for <SEP> and <CLS>
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model
            }
        }
    },
  "train_data_path":      "/home/gereka/data/nlp/Turkish/text_classification/hurriyet6c1k/hurriyet6c1k_train.csv",
  "validation_data_path": "/home/gereka/data/nlp/Turkish/text_classification/hurriyet6c1k/hurriyet6c1k_dev.csv",
  "test_data_path":       "/home/gereka/data/nlp/Turkish/text_classification/hurriyet6c1k/hurriyet6c1k_test.csv",
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
