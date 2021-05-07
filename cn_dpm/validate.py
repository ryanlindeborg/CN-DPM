from typing import Optional
from torch import Tensor
from .models import NdpmModel
from sequoia.settings.passive import ClassIncrementalSetting, PassiveEnvironment


def validate_model(config, model: NdpmModel, sequoia_env: PassiveEnvironment):
    assert len(model.ndpm.experts) > 1
    model.eval()
    observations: ClassIncrementalSetting.Observations
    rewards: Optional[ClassIncrementalSetting.Rewards]
    for step, (observations, rewards) in enumerate(sequoia_env):
        x: Tensor = observations.x
        t: Optional[Tensor] = observations.task_labels
        y: Optional[Tensor] = rewards.y if rewards is not None else None

        # TODO: Adapt this evaluation stuff:
        # NOTE: @lebrice There are many ways to do this, some easier than others:
        # - HARD: Adapt their evaluation loops
        # - EASY, not-optimal: Create a Dataset of the right type using the contents of
        # the environment.
        raise NotImplementedError("TODO")
        model.evaluate_model(sequoia_env, step, 'model')
        
    """
    Existing DataScheduler eval loop through datasets:
            for i, eval_dataset in enumerate(self.eval_datasets.values()):
            # NOTE: we assume that each task is a dataset in multi-dataset
            # episode
            eval_dataset.eval(
                model, writer, step, eval_title,
                task_index=(i if len(self.eval_datasets) > 1 else None)
            )
    """