from allennlp.commands.train import train_model_from_file


if __name__ == '__main__':
    train_model_from_file(
        parameter_filename = '/home/gereka/code/personal/nlp/experiments/text_classification/2021_05_11_002_basic_stanford_sentiment_treebank.jsonnet',
        serialization_dir  = '/home/gereka/outputs/nlp/text_classification/2021_05_11_002__2021_05_11_002_basic_stanford_sentiment_treebank',
        include_package    = ['agnlp.custom'],
    )
