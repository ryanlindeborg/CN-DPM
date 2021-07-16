from typing import Optional
from torch import Tensor
from .models import NdpmModel
from sequoia.settings.sl import ContinualSLSetting, SLEnvironment


def validate_model(config, model: NdpmModel, sequoia_env: SLEnvironment):
    # TODO: Why > instead of >=?
    assert len(model.ndpm.experts) >= 1
    model.eval()
    observations: ContinualSLSetting.Observations
    rewards: Optional[ContinualSLSetting.Rewards]
    model.ndpm.evaluate_model(sequoia_env)
        
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