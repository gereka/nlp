local bert_model = "bert-base-cased";
//local transformer_dim = 1024;


{
    "pytorch_seed": 4303,
    "dataset_reader": {
        "type": "fasttext_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model
        },
        //"lazy": true,
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model
            }
        }
    },
  "train_data_path":      "/home/gereka/data/nlp/yelp_review_polarity_csv/train.csv",
  "validation_data_path": "/home/gereka/data/nlp/yelp_review_polarity_csv/dev.csv",
  "test_data_path": 	  "/home/gereka/data/nlp/yelp_review_polarity_csv/test.csv",
    "model": {
//        "namespace": 'tags',
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
            "batch_size": 32,
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
        "cuda_device": -1
    },
  "evaluate_on_test": true
}
