import gym
import yaml
from typing import Tuple
from typing import ClassVar, Type

from sequoia.methods import Method
from sequoia.settings import Setting, Environment
from sequoia.settings.passive import PassiveSetting, PassiveEnvironment, ClassIncrementalSetting

from models.ndpm_model import NdpmModel

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
        image_size: Tuple[int, ...] = setting.observation_space.x.shape
        print(f"image_size: {image_size}")
        number_of_tasks = setting.nb_tasks
        print(f"Number of tasks: {number_of_tasks}")
        # TODO: Create the Model, using the image shape, the number of tasks (if that
        # makes sense), etc.
        # Create config from setting variable, to pass into model when initialize
        config = yaml.load(open(CNDPM_YAML_PATH), Loader=yaml.FullLoader)
        # TODO Set dimensions of input and output: x_c: 1
        # x_h: 28
        # x_w: 28
        # y_c: 10
        # TODO: Maybe delete and set all properties of model manually
        self.model = self.ModelType(
            #observation_space=setting.observation_space,
            #action_space=setting.action_space,
            #reward_space=setting.reward_space,
            config,
            writer: SummaryWriter
        )

    def fit(self, train_env: Environment, valid_env: Environment):
        """Called by the Setting to give the method data to train with.
        
        Might be called more than once before training is 'complete'.
        """
        raise NotImplementedError("TODO: Train the model on the data from the environments.") 
    
    def get_actions(self,
                    observations: ClassIncrementalSetting.Observations,
                    action_space: gym.Space) -> ClassIncrementalSetting.Actions:
        """ Get a batch of predictions (actions) for the given observations.
        returned actions must fit the action space.
        """
        pass


if __name__ == "__main__":
    from sequoia.settings.passive import ClassIncrementalSetting
    setting = ClassIncrementalSetting(dataset="mnist", nb_tasks=5)    
    method = CNDPM()

    results = setting.apply(method)
    print(results.summary())
