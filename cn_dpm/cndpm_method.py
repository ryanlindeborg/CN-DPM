import json
import os
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import gym
import torch
import yaml
from sequoia.methods import Method, register_method
from sequoia.settings import Environment, Setting
from sequoia.settings.sl import (ContinualSLSetting, PassiveEnvironment,
                                 SLSetting)
from simple_parsing import ArgumentParser, mutable_field
from simple_parsing.helpers import list_field
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.hparams import HyperParameters
from torch import Tensor

from cn_dpm.config import (DatasetConfig, DPMoEConfig, EvalConfig, ModelConfig,
                           SummaryConfig, TrainConfig)
from cn_dpm.models.ndpm_model import NdpmModel
from cn_dpm.train import train_model_with_sequoia_env
from cn_dpm.validate import validate_model

# Directory containing this source code.
SOURCE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# Directory containing the 'configs'.
CONFIGS_DIR = SOURCE_DIR / "cn_dpm" / "configs"
CNDPM_YAML_PATH = CONFIGS_DIR / "cndpm.yaml"


@dataclass
class HParams(HyperParameters, FlattenedAccess):
    """ Hyper-parameters of the CN-DPM model. """

    learning_rate: float = 3e-4

    # Denotes whether to use CPU instead of CUDA device
    disable_cuda: bool = False
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    dataset: DatasetConfig = mutable_field(DatasetConfig)
    model: ModelConfig = mutable_field(ModelConfig)
    dpmoe: DPMoEConfig = mutable_field(DPMoEConfig)
    train: TrainConfig = mutable_field(TrainConfig)
    eval: EvalConfig = mutable_field(EvalConfig)
    summary: SummaryConfig = mutable_field(SummaryConfig)

    def __post_init__(self):
        if self.disable_cuda:
            self.device = torch.device("cpu")


class CNDPM(Method, target_setting=ContinualSLSetting):
    """ A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning

    https://arxiv.org/abs/2001.00689
    """
    # Note: We put this as a class attribute in case subclasses (or tests) wanted to
    # overwrite this value.
    ModelType: ClassVar[Type[NdpmModel]] = NdpmModel

    def __init__(self, hparams: HParams):
        # The hparams here consists of the model hyperparameters
        self.hparams = hparams
        self.device = self.hparams.device
        # We will create this when `configure` is called, before training.
        self.model: NdpmModel

    def configure(self, setting: ContinualSLSetting):
        """Configures this method before it gets applied on the given Setting.

        NOTE: This will be called by the Setting, you don't need to call this yourself.

        Args:
            setting (SettingType): The setting the method will be evaluated on.
        """
        self.setting = setting
        print(f"Observations space: {setting.observation_space}")
        # Observation space is tuple consisting of number of channels, height of image, width of image
        image_size: Tuple[int, ...] = setting.observation_space.x.shape
        print(f"image_size: {image_size}")
        x_c, x_h, x_w = image_size
        # NOTE: @lebrice Why are these being set in the dataclass?
        self.hparams["x_c"] = x_c
        self.hparams["x_h"] = x_h
        self.hparams["x_w"] = x_w

        number_of_tasks = setting.nb_tasks
        print(f"Number of tasks: {number_of_tasks}")

        # Fetch output size from action space size
        self.hparams["y_c"] = setting.action_space.n

        self.model = self.ModelType(self.hparams)
        self.model.to(self.device)

    def fit(self, train_env: Environment, valid_env: Environment):
        """Called by the Setting to give the method data to train with.

        Might be called more than once before training is 'complete'.
        """
        # Train loop
        train_model_with_sequoia_env(self.hparams, self.model, train_env)
        # Validaton loop
        # TODO: Fix the validation loop (see `validate_model` function)
        validate_model(self.hparams, self.model, valid_env)

    def get_actions(
        self,
        observations: ContinualSLSetting.Observations,
        action_space: gym.Space,
    ) -> ContinualSLSetting.Actions:
        """ Get a batch of predictions (actions) for the given observations.
        returned actions must fit the action space.
        """
        self.model.eval()
        x: Tensor = observations.x
        x = x.to(device=self.device)
        with torch.no_grad():
            logits = self.model(x)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = None) -> None:
        """Add the command-line arguments for this Method to the given parser. """
        prefix = dest + "." if dest else ""
        parser.add_arguments(HParams, f"{prefix}hparams")
        parser.add_argument(
            f"--{prefix}load_path",
            type=str,
            default=None,
            help="If given, the HyperParameters are read from the given file instead of from the command-line."
        )

    @classmethod
    def from_argparse_args(cls, args, dest: str = None) -> "CNDPM":
        args = getattr(args, dest) if dest else args
        load_path: str = args.load_path
        hparams: HParams = args.hparams
        if load_path:
            hparams = HParams.load_json(load_path)
        print(f"Logging hparams: {hparams}")
        return CNDPM(hparams=hparams)


if __name__ == "__main__":
    # setting = ContinualSLSetting(dataset="mnist", nb_tasks=5)
    from sequoia.settings.sl import ClassIncrementalSetting
    setting = ContinualSLSetting(dataset="cifar10", nb_tasks=5)
    # setting = ClassIncrementalSetting(dataset="fashionmnist", nb_tasks=5)

    parser = ArgumentParser(add_dest_to_option_strings=True)
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="If given, the HyperParameters are read from the given file instead of from the command-line."
    )
    parser.add_arguments(HParams, dest="hparams")
    args = parser.parse_args()

    load_path: str = args.load_path
    if load_path is None:
        hparams: HParams = args.hparams
    else:
        hparams = HParams.load_json(load_path)
    print(f"Logging hparams: {hparams}")

    method = CNDPM(hparams)

    results = setting.apply(method)
    print(results.summary())
