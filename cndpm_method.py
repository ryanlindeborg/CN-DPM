import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Tuple, Type

import gym
import torch
import yaml
from sequoia.methods import Method
from sequoia.settings import Environment, Setting
from sequoia.settings.passive import (
    ClassIncrementalSetting,
    PassiveEnvironment,
    PassiveSetting,
)
from simple_parsing.helpers.hparams import HyperParameters

from cn_dpm.models.ndpm_model import NdpmModel
from cn_dpm.train import train_model_with_sequoia_env
from cn_dpm.validate import validate_model

# Directory containing this source code.
SOURCE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# Directory containing the 'configs'.
CONFIGS_DIR = SOURCE_DIR / "cn_dpm" / "configs"
CNDPM_YAML_PATH = CONFIGS_DIR / "cndpm.yaml"


# TODO: (medium-priority) Create a `Config` or `HParams` dataclass containing the
# hyper-parameters of the model, which are currently all defined inside the config
# files.

# Then, when creating the model from cn_dpm, we could convert this `HParams` object into
# a dict with either `hparams.to_dict()` or `dataclasses.asdict(hparams)`, and then pass
# the dict to the constructor, just as before.

# @lebrice The reason I think this is a good idea is that it would make it much
# easier to see what the hyper-parameters are, and also make it easier for others to
# reuse/extend this model / method and inherit their hyper-parameters.

# NOTE: (@lebrice) You can (and might probably have to) create multiple Config
# dataclasses, one for each group of related parameters! For example, one dataclass for
# the `OptimizerConfig`, another for the `TrainConfig`, etc etc. These config classes
# can also be nested within a larger `ModelConfig` (or something similar) by using them
# as a field of the parent class.

# Actually, funily enough, I think I started doing this myself a while back! You can use
# this as inspiration if you want:
# https://github.com/lebrice/SimpleParsing/blob/master/test/utils/test_flattened.py


@dataclass
class HParams(HyperParameters):
    """ Hyper-parameters of the CN-DPM model. """

    # We could also pass a `HParams` object to the Model constructor, rather than a
    # dictionary, and then just add a few methods like this to make it behave like
    # a dict. Its probably easier to just convert this object to a dict though.
    # def __getitem__(self, key: str):
    #     return getattr(self, key)

    # def __setitem__(self, key: str, value: Any) -> None:
    #     setattr(self, key, value)


class CNDPM(Method, target_setting=ClassIncrementalSetting):
    """ A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning

    https://arxiv.org/abs/2001.00689
    """

    ModelType: ClassVar[Type[NdpmModel]] = NdpmModel

    def __init__(self, learning_rate: float = 3e-4):
        self.learning_rate = learning_rate

        # We will create this when `configure` is called, before training.
        self.model: NdpmModel

    def configure(self, setting: ClassIncrementalSetting):
        """Configures this method before it gets applied on the given Setting.

        NOTE: This will be called by the Setting, you don't need to call this yourself.

        Args:
            setting (SettingType): The setting the method will be evaluated on.
        """
        print(f"Observations space: {setting.observation_space}")
        # Load config, to pass into model when initialize
        config = yaml.load(open(CNDPM_YAML_PATH), Loader=yaml.FullLoader)
        # Observation space is tuple consisting of number of channels, height of image, width of image
        image_size: Tuple[int, ...] = setting.observation_space.x.shape
        print(f"image_size: {image_size}")
        x_c, x_h, x_w = image_size
        config["x_c"] = x_c
        config["x_h"] = x_h
        config["x_w"] = x_w

        number_of_tasks = setting.nb_tasks
        print(f"Number of tasks: {number_of_tasks}")
        config["y_c"] = number_of_tasks

        self.model = self.ModelType(config,)
        self.model.to(config["device"])

    def fit(self, train_env: Environment, valid_env: Environment):
        """Called by the Setting to give the method data to train with.

        Might be called more than once before training is 'complete'.
        """
        config = yaml.load(open(CNDPM_YAML_PATH), Loader=yaml.FullLoader)
        # data_scheduler = DataScheduler(config)

        # Train loop
        train_model_with_sequoia_env(config, self.model, train_env)
        # Validaton loop
        # TODO: Fix the validation loop (see `validate_model` function)
        # validate_model(config, self.model, valid_env)

    def get_actions(
        self,
        observations: ClassIncrementalSetting.Observations,
        action_space: gym.Space,
    ) -> ClassIncrementalSetting.Actions:
        """ Get a batch of predictions (actions) for the given observations.
        returned actions must fit the action space.
        """
        self.model.eval()
        observations = observations.to(device=self.device)
        with torch.no_grad():
            logits = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = "") -> None:
        """Add the command-line arguments for this Method to the given parser.

        Parameters
        ----------
        parser : ArgumentParser
            The ArgumentParser.
        dest : str, optional
            The 'base' destination where the arguments should be set on the
            namespace, by default empty, in which case the arguments can be at
            the "root" level on the namespace.
        """
        prefix = f"{dest}." if dest else ""
        parser.add_argument(f"--{prefix}log_dir", type=str, default="logs")


if __name__ == "__main__":
    setting = ClassIncrementalSetting(dataset="mnist", nb_tasks=5)
    method = CNDPM()

    results = setting.apply(method)
    print(results.summary())
