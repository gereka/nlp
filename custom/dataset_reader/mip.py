from typing import Dict
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer as WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("mip_dataset_reader2")
class MIPDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the venue of the paper.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_index: int=None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_index = label_index

    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        
        with open(cached_path(file_path), "r") as dataset_file:
            while True:
                line = dataset_file.readline()
                if not line: break
                    
                elems = line.split("~")
                line_label = elems[0]
                line_complaint = elems[1]                

                line_complaint = line_complaint.strip("\n")
                line_label = line_label.strip("\n")

                if self._label_index==-1: #multi-output joint distribution
                    yield self.text_to_instance(line_complaint, line_label)
                else: #single output. label_index determines which output
                    line_labels = line_label.split(";")
                    yield self.text_to_instance(line_complaint, line_labels[self._label_index])
                

    @overrides
    #def text_to_instance(self, title: str, abstract: str, venue: str = None) -> Instance:  # type: ignore
    def text_to_instance(self, complaint: str, product: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_complaint = self._tokenizer.tokenize(complaint)  
        if len(tokenized_complaint) > 500: tokenized_complaint = tokenized_complaint[:500]
        complaint_field = TextField(tokenized_complaint, self._token_indexers)        
        fields = {'tokens': complaint_field}
        if product is not None:
            fields['label'] = LabelField(product)
        return Instance(fields)

    # @classmethod
    # def from_params(cls, params: Params) -> 'MIPDatasetReader':
    #     lazy = params.pop('lazy', False)
    #     tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
    #     token_indexers = dict_from_params(TokenIndexer, params.pop('token_indexers', {}))
    #     label_index = params.pop_int('label_index', -1)
    #     params.assert_empty(cls.__name__)
    #     return cls(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers, label_index=label_index)
    #

#fmg 18.09.2018: this does not exist in allennlp 0.6.1 and I can't find a replacement for it in the new version. Copying from allennlp repo old version.
def dict_from_params(cls, params: Params) -> 'Dict[str, TokenIndexer]':  # type: ignore
    """
    We typically use ``TokenIndexers`` in a dictionary, with each ``TokenIndexer`` getting a
    name.  The specification for this in a ``Params`` object is typically ``{"name" ->
    {indexer_params}}``.  This method reads that whole set of parameters and returns a
    dictionary suitable for use in a ``TextField``.
    Because default values for token indexers are typically handled in the calling class to
    this and are based on checking for ``None``, if there were no parameters specifying any
    token indexers in the given ``params``, we return ``None`` instead of an empty dictionary.
    """
    token_indexers = {}
    for name, indexer_params in params.items():
        token_indexers[name] = cls.from_params(indexer_params)
    if token_indexers == {}:
        token_indexers = None
    return token_indexers
