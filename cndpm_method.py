import os
from dataclasses import dataclass, field, asdict, InitVar
from pathlib import Path
from typing import ClassVar, Tuple, Type, Dict, Any, Optional, List

import gym
import torch
from torch import Tensor
import yaml
import json
from sequoia.methods import Method, register_method
from sequoia.settings import Environment, Setting
from sequoia.settings.passive import (
    ClassIncrementalSetting,
    PassiveEnvironment,
    PassiveSetting,
)

from simple_parsing.helpers import list_field
from simple_parsing import ArgumentParser, mutable_field
from simple_parsing.helpers.hparams import HyperParameters
from simple_parsing.helpers.flatten import FlattenedAccess

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

@dataclass()
class ObjectConfig:
    """Configuration for a generic Object with a type and some kwargs."""
    type: str = ""
    options: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


@dataclass
class DatasetConfig:
    """Dataset Config."""
    sleep_batch_size: int = 50
    sleep_num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "ndpm_model"
    g: str = "mlp_sharing_vae"
    d: str = "mlp_sharing_classifier"
    disable_d: bool = False
    vae_nf_base: int = 64
    vae_nf_ext: int = 16
    cls_nf_base: int = 64
    cls_nf_ext: int = 16
    z_dim: int = 16
    z_samples: int = 16

    pretrained_init: Optional[Dict] = None
    precursor_conditioned_decoder: Optional[bool] = None
    recon_loss: str = "gaussian"
    x_log_var_param: Optional[int] = 0
    learn_x_log_var: Optional[bool] = False
    classifier_chill: float = 0.01


@dataclass
class DPMoEConfig:
    """Configuration of the Dirichlet Process Mixture of Experts Model. """
    log_alpha: int = -400
    stm_capacity: int = 500
    sleep_val_size: int = 0
    stm_erase_period: int = 0

    sleep_step_g: int = 8000
    sleep_step_d: int = 2000
    sleep_summary_step: int = 500

    known_destination: Optional[List[int]] = None
    update_min_usage: float = 0.1
    send_to_stm_always: Optional[bool] = None


@dataclass
class TrainConfig:
    """ Training Configuration. """
    weight_decay: float = 0.00001
    implicit_lr_decay: bool = False
    optimizer_g: ObjectConfig = ObjectConfig(type="Adam", options={"lr": 0.0004})
    # optimizer_d: Optional[ObjectConfig] = ObjectConfig(type="Adam", options={"lr": 0.0001})
    optimizer_d: ObjectConfig = ObjectConfig(type="Adam", options={"lr": 0.0001})
    lr_scheduler_g: ObjectConfig = ObjectConfig(type="MultiStepLR", options={"milestones": [1], "gamma": 1.0})
    # lr_scheduler_d: Optional[ObjectConfig] = ObjectConfig(type="MultiStepLR", options={"milestones": [1], "gamma": 1.0})
    lr_scheduler_d: ObjectConfig = ObjectConfig(type="MultiStepLR", options={"milestones": [1], "gamma": 1.0})
    clip_grad: ObjectConfig = ObjectConfig(type="value", options={"clip_value": 0.5})


@dataclass
class EvalConfig:
    """ Eval configuration. """
    eval_d: bool = True
    eval_g: bool = False
    eval_t: Optional[bool] = False

@dataclass
class SummaryConfig:
    """ Summary configuration """
    summary_step: int = 250
    eval_step: int = 250
    summarize_samples: bool = False
    sample_grid: List[int] = list_field(10, 10)


@dataclass
class HParams(HyperParameters, FlattenedAccess):
    """ Hyper-parameters of the CN-DPM model. """

    # Denotes whether to use CPU instead of CUDA device
    disable_cuda: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # # This config name (in conjunction with episode parameter) describes the setting in
    # # which the CNDPM model is run, per the original repo labels (per original repo
    # # labels)
    # cndpm_config: Optional[str] = None
    episode: Optional[str] = None  # This config name (in conjunction with episode parameter) describes the setting in which the CNDPM model is run, per the original repo labels (per original repo labels)

    dataset: DatasetConfig = mutable_field(DatasetConfig)
    model: ModelConfig = mutable_field(ModelConfig)
    dpmoe: DPMoEConfig = mutable_field(DPMoEConfig)
    train: TrainConfig = mutable_field(TrainConfig)
    eval: EvalConfig = mutable_field(EvalConfig)
    summary: SummaryConfig = mutable_field(SummaryConfig)


@register_method
class CNDPM(Method, target_setting=ClassIncrementalSetting):
    """ A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning

    https://arxiv.org/abs/2001.00689
    """

    ModelType: ClassVar[Type[NdpmModel]] = NdpmModel

    def __init__(self, hparams: HParams, learning_rate: float = 3e-4):
        # The cn_dpm_config here consists of the model hyperparameters
        eval_steps = hparams.summary.eval_step
        eval_steps = hparams.eval_step
        
        self.cn_dpm_config = hparams
        if self.cn_dpm_config.disable_cuda:
            self.cn_dpm_config.device = "cpu"
        self.learning_rate = learning_rate
        self.device = self.cn_dpm_config.device

        # We will create this when `configure` is called, before training.
        self.model: NdpmModel

    def configure(self, setting: ClassIncrementalSetting):
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
        self.cn_dpm_config["x_c"] = x_c
        self.cn_dpm_config["x_h"] = x_h
        self.cn_dpm_config["x_w"] = x_w

        number_of_tasks = setting.nb_tasks
        print(f"Number of tasks: {number_of_tasks}")

        # Fetch output size from action space size
        self.cn_dpm_config["y_c"] = setting.action_space.n

        self.model = self.ModelType(self.cn_dpm_config)
        self.model.to(self.cn_dpm_config["device"])

    def fit(self, train_env: Environment, valid_env: Environment):
        """Called by the Setting to give the method data to train with.

        Might be called more than once before training is 'complete'.
        """
        # Train loop
        train_model_with_sequoia_env(self.cn_dpm_config, self.model, train_env)
        # Validaton loop
        # TODO: Fix the validation loop (see `validate_model` function)
        validate_model(self.cn_dpm_config, self.model, valid_env)

    def get_actions(
        self,
        observations: ClassIncrementalSetting.Observations,
        action_space: gym.Space,
    ) -> ClassIncrementalSetting.Actions:
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




if __name__ == "__main__":
    setting = ClassIncrementalSetting(dataset="mnist", nb_tasks=5)

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
