import pytest
from typing import ClassVar, List, Optional, Type
from sequoia.settings import Setting
from sequoia.settings.passive import ClassIncrementalSetting, TaskIncrementalSetting

from cn_dpm.cndpm_method import CNDPM, HParams, DatasetConfig, ModelConfig, DPMoEConfig, TrainConfig, EvalConfig, SummaryConfig
from cn_dpm.models.ndpm_model import NdpmModel

class TestCNDPMMethod:
    Method: ClassVar[Type[CNDPM]] = CNDPM

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "setting_type",
        [
            ClassIncrementalSetting,
            TaskIncrementalSetting,
        ]
    )
    def test_cndpm_method_on_settings(
            self,
            setting_type: Type[Setting]
    ):
        setting = setting_type(dataset="mnist", nb_tasks=5)
        hparams = HParams(
            dataset=DatasetConfig(),
            model=ModelConfig(),
            dpmoe=DPMoEConfig(),
            train=TrainConfig(),
            eval=EvalConfig(),
            summary=SummaryConfig())
        method = self.Method(hparams)

        results = setting.apply(method)
        print(results.summary())