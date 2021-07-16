"""
TODO: (medium-priority) Create a `Config` or `HParams` dataclass containing the
hyper-parameters of the model, which are currently all defined inside the config
files.

Then, when creating the model from cn_dpm, we could convert this `HParams` object into
a dict with either `hparams.to_dict()` or `dataclasses.asdict(hparams)`, and then pass
the dict to the constructor, just as before.

@lebrice The reason I think this is a good idea is that it would make it much
easier to see what the hyper-parameters are, and also make it easier for others to
reuse/extend this model / method and inherit their hyper-parameters.

NOTE: (@lebrice) You can (and might probably have to) create multiple Config
dataclasses, one for each group of related parameters! For example, one dataclass for
the `OptimizerConfig`, another for the `TrainConfig`, etc etc. These config classes
can also be nested within a larger `ModelConfig` (or something similar) by using them
as a field of the parent class.

Actually, funily enough, I think I started doing this myself a while back! You can use
this as inspiration if you want:
https://github.com/lebrice/SimpleParsing/blob/master/test/utils/test_flattened.py
"""
from dataclasses import dataclass
from simple_parsing.helpers import list_field, field
from simple_parsing import ArgumentParser, mutable_field
from simple_parsing.helpers.flatten import FlattenedAccess
from typing import Dict, Any, Optional, List


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
    x_log_var_param: int = 0
    learn_x_log_var: bool = False
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
    optimizer_d: ObjectConfig = ObjectConfig(type="Adam", options={"lr": 0.0001})
    lr_scheduler_g: ObjectConfig = ObjectConfig(type="MultiStepLR", options={"milestones": [1], "gamma": 1.0})
    lr_scheduler_d: ObjectConfig = ObjectConfig(type="MultiStepLR", options={"milestones": [1], "gamma": 1.0})
    clip_grad: ObjectConfig = ObjectConfig(type="value", options={"clip_value": 0.5})


@dataclass
class EvalConfig:
    """ Eval configuration. """
    eval_d: bool = True
    eval_g: bool = False
    eval_t: bool = False

@dataclass
class SummaryConfig:
    """ Summary configuration """
    summary_step: int = 250
    eval_step: int = 250
    summarize_samples: bool = False
    sample_grid: Optional[List[int]] = None
    # sample_grid: List[int] = list_field(10, 10)
