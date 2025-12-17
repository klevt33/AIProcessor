from logging import Logger

from data_loader import DataLoader
from fine_tuning_state import FinetuneJobState
from fine_tuning_utils import FineTuningUtils

from config import Config


class Context:
    config: Config | None = None
    logger: Logger | None = None
    ft_utils: FineTuningUtils | None = None
    data_loader: DataLoader | None = None
    fine_tuning_utils: FineTuningUtils | None = None
    fine_tuning_state: FinetuneJobState | None = None


context = Context()
