import gym
import yaml
from typing import Tuple
from typing import ClassVar, Type
from tensorboardX import SummaryWriter
from argparse import ArgumentParser

from sequoia.methods import Method
from sequoia.settings import Setting, Environment
from sequoia.settings.passive import PassiveSetting, PassiveEnvironment, ClassIncrementalSetting

from models.ndpm_model import NdpmModel
from train import train_model
from validate import validate_model

CNDPM_YAML_PATH = './sequoia/methods/cn_dpm/configs/cndpm.yaml'

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

        writer = SummaryWriter(config['log_dir'])
        self.model = self.ModelType(
            config,
            writer,
        )
        self.model.to(config['device'])

    def fit(self, train_env: Environment, valid_env: Environment):
        """Called by the Setting to give the method data to train with.
        
        Might be called more than once before training is 'complete'.
        """
        config = yaml.load(open(CNDPM_YAML_PATH), Loader=yaml.FullLoader)
        # data_scheduler = DataScheduler(config)

        # Train loop
        train_model(config, self.model, data_scheduler, self.model.writer)
        # Validaton loop

        # raise NotImplementedError("TODO: Train the model on the data from the environments.")
    
    def get_actions(self,
                    observations: ClassIncrementalSetting.Observations,
                    action_space: gym.Space) -> ClassIncrementalSetting.Actions:
        """ Get a batch of predictions (actions) for the given observations.
        returned actions must fit the action space.
        """
        pass

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
    from sequoia.settings.passive import ClassIncrementalSetting
    setting = ClassIncrementalSetting(dataset="mnist", nb_tasks=5)
    method = CNDPM()

    results = setting.apply(method)
    print(results.summary())
