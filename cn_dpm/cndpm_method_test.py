from typing import ClassVar, List, Optional, Type

import pytest
from sequoia.common.config import Config
from sequoia.methods.method_test import MethodTests
from sequoia.settings import Setting
from sequoia.settings.sl import ContinualSLSetting, TaskIncrementalSLSetting

from cn_dpm.cndpm_method import (CNDPM, DatasetConfig, DPMoEConfig, EvalConfig,
                                 HParams, ModelConfig, SummaryConfig,
                                 TrainConfig)
from cn_dpm.models.ndpm_model import NdpmModel


class TestCNDPMMethod(MethodTests):
    """ Tests for the CN-DPM Method.
    
    The main test of interest is `test_debug`, which is implemented in the MethodTests
    class.
    """
    
    Method: ClassVar[Type[CNDPM]] = CNDPM

    @pytest.mark.skip(reason="A bit too long to run")
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "setting_type",
        [
            ContinualSLSetting,
            TaskIncrementalSLSetting,
        ]
    )
    def test_cndpm_method_on_settings(
            self,
            setting_type: Type[Setting]
    ):
        setting = setting_type(dataset="mnist", nb_tasks=5)
        hparams = HParams()
        method = self.Method(hparams)

        results = setting.apply(method)
        print(results.summary())

    @classmethod
    @pytest.fixture
    def method(cls, config: Config) -> CNDPM:
        """ Fixture that returns the Method instance to use when testing/debugging.

        Needs to be implemented when creating a new test class (to generate tests for a
        new method).
        """
        debug_hparams = HParams(device=config.device)
        if config.debug:
            # TODO: Set the parameters for a debugging run on a short setting!
            # (Need runs to be shorter than 30 secs per setting!)
            debug_hparams.dpmoe = DPMoEConfig(
                stm_capacity=50,
                send_to_stm_always=True,
                sleep_step_g=100,
                sleep_step_d=100,
            )
            # Just for reference:
            # @dataclass
            # class DPMoEConfig:
            #     """Configuration of the Dirichlet Process Mixture of Experts Model. """
            #     log_alpha: int = -400
            #     stm_capacity: int = 500
            #     sleep_val_size: int = 0
            #     stm_erase_period: int = 0

            #     sleep_step_g: int = 8000
            #     sleep_step_d: int = 2000
            #     sleep_summary_step: int = 500

            #     known_destination: Optional[List[int]] = None
            #     update_min_usage: float = 0.1
            #     send_to_stm_always: Optional[bool] = None         

        return cls.Method(debug_hparams)

    def validate_results(self,
                         setting: Setting,
                         method: CNDPM,
                         results: Setting.Results,
                         ) -> None:
        assert results
        assert results.objective
        # TODO: Add reasonable bounds on expected performance
