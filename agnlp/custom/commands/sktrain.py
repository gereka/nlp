import argparse
from os import PathLike
from typing import Any, Dict, List, Optional, Union
from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params
from allennlp.models.model import Model

@Subcommand.register("sktrain")
class SKTrain(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train using sci-kit learn models."""
        subparser = parser.add_parser(self.name, description=description, help="Train a sci-kit learn model.")

        subparser.add_argument(
            "param_path", type=str, help="path to parameter file describing the model to be trained"
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the model and its logs",
        )

        # subparser.add_argument(
        #     "-r",
        #     "--recover",
        #     action="store_true",
        #     default=False,
        #     help="recover training from the state in serialization_dir",
        # )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--node-rank", type=int, default=0, help="rank of this node in the distributed setup"
        )

        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "do not train a model, but create a vocabulary, show dataset statistics and "
                "other training information"
            ),
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=sktrain_model_from_args)

        return subparser

def sktrain_model_from_args(args: argparse.Namespace):
    """
    Just converts from an `argparse.Namespace` object to string paths.
    """
    sktrain_model_from_file(
        parameter_filename=args.param_path,
        serialization_dir=args.serialization_dir,
        overrides=args.overrides,
#        recover=args.recover,
        force=args.force,
        node_rank=args.node_rank,
        include_package=args.include_package,
        dry_run=args.dry_run,
        file_friendly_logging=args.file_friendly_logging,
    )



def sktrain_model_from_file(
    parameter_filename: Union[str, PathLike],
    serialization_dir: Union[str, PathLike],
    overrides: Union[str, Dict[str, Any]] = "",
#    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: List[str] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
    return_model: Optional[bool] = None,
) -> Optional[Model]:    

    params = Params.from_file(parameter_filename, overrides)
    return sktrain_model(
        params=params,
        serialization_dir=serialization_dir,
#        recover=recover,
        force=force,
        node_rank=node_rank,
        include_package=include_package,
        dry_run=dry_run,
        file_friendly_logging=file_friendly_logging,
        return_model=return_model,
    )

def sktrain_model(
    params: Params,
    serialization_dir: Union[str, PathLike],
#    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: List[str] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
    return_model: Optional[bool] = None,
) -> Optional[Model]:

    print("Hello World")
