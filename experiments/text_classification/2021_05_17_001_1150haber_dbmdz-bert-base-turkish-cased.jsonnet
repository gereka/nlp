local bert_model = "/home/gereka/data/models/dbmdz-bert-base-turkish-cased/";

local pytorch_seed = std.parseInt(std.extVar('pytorch_seed'));
local dropout = std.parseJson(std.extVar('dropout'));
local learning_rate = std.parseJson(std.extVar('learning_rate'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));


{
    "pytorch_seed": pytorch_seed,
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
  "train_data_path":      "/home/gereka/data/nlp/Turkish/text_classification/1150haber/1150haber_train.csv",
  "validation_data_path": "/home/gereka/data/nlp/Turkish/text_classification/1150haber/1150haber_dev.csv",
  "test_data_path":       "/home/gereka/data/nlp/Turkish/text_classification/1150haber/1150haber_test.csv",
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
       "dropout": dropout
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
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        "cuda_device": 0,
	"callbacks": [
	    {
	        type: 'optuna_pruner',
            }
        ],
    },
  "evaluate_on_test": true
}
