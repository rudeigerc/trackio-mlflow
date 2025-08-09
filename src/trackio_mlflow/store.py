import logging
import os
import random
from typing import TYPE_CHECKING
from uuid import uuid4

import trackio
from mlflow.entities import LifecycleStage, Run, RunData, RunInfo, RunStatus
from mlflow.store.tracking.abstract_store import AbstractStore
from trackio import context_vars
from typing_extensions import override

if TYPE_CHECKING:
    from mlflow.entities import Metric, Param, RunTag


logger = logging.getLogger(__name__)

_MLFLOW_END_RUN_STATUSES = [
    RunStatus.FINISHED,
    RunStatus.FAILED,
    RunStatus.KILLED,
]


class TrackioStore(AbstractStore):
    _run_map: dict[str, trackio.Run]
    _trackio_project: str
    _space_id: str | None

    def __init__(self, store_uri: str | None, artifact_uri: str | None):
        self._run_map = {}
        self._trackio_project = os.getenv(
            "TRACKIO_PROJECT", f"trackio-{random.randint(100000, 999999)}"
        )
        self._space_id = os.getenv("TRACKIO_SPACE_ID", None)

        super().__init__()

    @override
    def create_run(
        self,
        experiment_id: str,
        user_id: str,
        start_time: int,
        tags: list["RunTag"],
        run_name: str,
    ) -> Run:
        current_run = context_vars.current_run.get()

        if current_run is None:
            current_run = trackio.init(
                project=self._trackio_project,
                name=run_name,
                space_id=self._space_id,
            )
            self._run_map[current_run.name] = current_run

        return Run(
            run_info=RunInfo(
                run_id=current_run.name,
                run_uuid=uuid4().hex,
                experiment_id=experiment_id,
                status=RunStatus.RUNNING,
                user_id="",
                start_time=start_time,
                end_time=None,
                lifecycle_stage=LifecycleStage.ACTIVE,
                run_name=current_run.name,
                artifact_uri="file:///tmp/",
            ),
            run_data=RunData(),
        )

    @override
    def update_run_info(
        self,
        run_id: str,
        run_status: RunStatus,
        end_time: int,
        run_name: str,
    ) -> RunInfo:
        if (
            run_status in _MLFLOW_END_RUN_STATUSES
            and self._run_map[run_id].name is context_vars.current_run.get().name
        ):
            trackio.finish()

        return RunInfo(
            run_id=run_id,
            run_uuid=uuid4().hex,
            experiment_id=uuid4().hex,
            status=run_status,
            end_time=end_time,
            user_id="",
            start_time=1,
            lifecycle_stage=LifecycleStage.ACTIVE,
            run_name=run_name,
            artifact_uri="file:///tmp/",
        )

    @override
    def log_batch(
        self,
        run_id: str,
        metrics: list["Metric"],
        params: list["Param"],
        tags: list["RunTag"],
    ) -> None:
        step = None
        metrics_dict = {}

        if len(metrics) > 0:
            for metric in metrics:
                metrics_dict[metric.key] = metric.value
                if step is None and hasattr(metric, "step") and metric.step is not None:
                    step = metric.step

            current_run = self._run_map[run_id]
            current_run.config.update(
                {param.key: param.value for param in params}, allow_val_change=True
            )
            current_run.log(metrics=metrics_dict, step=step)

    @override
    def get_run(self, run_id: str) -> Run:
        current_run = self._run_map[run_id]

        return Run(
            run_info=RunInfo(
                run_id=current_run.name,
                run_uuid=uuid4().hex,
                experiment_id=uuid4().hex,
                status=RunStatus.RUNNING,
                user_id="",
                start_time=1,
                end_time=2,
                lifecycle_stage=LifecycleStage.ACTIVE,
                run_name=current_run.name,
                artifact_uri="file:///tmp/",
            ),
            run_data=RunData(),
        )
