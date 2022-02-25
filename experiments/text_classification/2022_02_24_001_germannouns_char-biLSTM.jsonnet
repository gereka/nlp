{
    "pytorch_seed": 4304,
    "dataset_reader": {
        "type": "csvreader",
        "tokenizer": {
            "type": "character",
        },
        "token_indexers": {
	    "char" : {
	        "type": "single_id",
	    }
	    
	},
     
    },
  "train_data_path": "/home/gereka/data/nlp/German/text_classification/nouns_v2_train.csv",
  "validation_data_path": "/home/gereka/data/nlp/German/text_classification/nouns_v2_dev.csv",
  "test_data_path": "/home/gereka/data/nlp/German/text_classification/nouns_v2_test.csv",
    "model": {
//        "namespace": 'tags',
        "type": "basic_classifier",
        "text_field_embedder": {
            "token_embedders": {
                "char": {
		    "type" : "embedding",
                    "embedding_dim": 100,
                }
            }
        },
        "seq2vec_encoder": {
           "type": "lstm",
	   "num_layers": 5,
           "input_size": 100,
	   "hidden_size": 100,
	   "bidirectional": true,
//           "requires_grad": true,
	   "dropout": 0.5,
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["tokens"],
            "batch_size": 32,
        }
    },
    "trainer": {
        "num_epochs": 2500,
        "patience": 10,
        "validation_metric": "+accuracy",
//        "learning_rate_scheduler": {
//            "type": "slanted_triangular",
//            "num_steps_per_epoch": 3088,
//            "cut_frac": 0.06
//        },
//	  "learning_rate_scheduler": "constant",
//	"learning_rate_scheduler": { //couldn't get it to work
//            "type": "cosine",
//            "t_initial": 5,
//            "t_mul": 0.9,
//            "eta_min": 1e-12,
//            "eta_mul": 0.8,
//            "last_epoch": 10,
//      },
          "learning_rate_scheduler": {
              "type": "cosine_hard_restarts_with_warmup",
              "num_warmup_steps": 100,
	      "num_training_steps": 10000,
          },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-2,
        },
        "cuda_device": 0
    },
  "evaluate_on_test": true
}
