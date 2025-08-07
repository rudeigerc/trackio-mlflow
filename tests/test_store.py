import unittest
from unittest.mock import Mock, patch

from mlflow.entities import (
    LifecycleStage,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunStatus,
    RunTag,
)

from trackio_mlflow.store import TrackioStore


class TestTrackioStore(unittest.TestCase):
    def setUp(self) -> None:
        self.store = TrackioStore(store_uri="trackio://", artifact_uri=None)
        self.experiment_id = "test_experiment"
        self.user_id = "test_user"
        self.start_time = 1234567890
        self.run_name = "test_run"
        self.tags = [RunTag(key="test_key", value="test_value")]

    @patch("trackio_mlflow.store.context_vars")
    @patch("trackio_mlflow.store.trackio")
    def test_create_run_with_no_current_run(
        self, mock_trackio: Mock, mock_context_vars: Mock
    ) -> None:
        """Test creating a run when no current run exists."""
        mock_context_vars.current_run.get.return_value = None
        mock_run = Mock()
        mock_run.name = "test_run_id"
        mock_trackio.init.return_value = mock_run

        result = self.store.create_run(
            experiment_id=self.experiment_id,
            user_id=self.user_id,
            start_time=self.start_time,
            tags=self.tags,
            run_name=self.run_name,
        )

        mock_trackio.init.assert_called_once_with(project="mlflow")
        self.assertIsInstance(result, Run)
        self.assertEqual(result.info.run_id, "test_run_id")
        self.assertEqual(result.info.status, RunStatus.RUNNING)
        self.assertEqual(result.info.lifecycle_stage, LifecycleStage.ACTIVE)
        self.assertIn("test_run_id", self.store._run_map)

    @patch("trackio_mlflow.store.context_vars")
    def test_create_run_with_existing_current_run(
        self, mock_context_vars: Mock
    ) -> None:
        """Test creating a run when a current run already exists."""
        mock_run = Mock()
        mock_run.name = "existing_run_id"
        mock_context_vars.current_run.get.return_value = mock_run

        result = self.store.create_run(
            experiment_id=self.experiment_id,
            user_id=self.user_id,
            start_time=self.start_time,
            tags=self.tags,
            run_name=self.run_name,
        )

        self.assertIsInstance(result, Run)
        self.assertEqual(result.info.run_id, "existing_run_id")
        self.assertEqual(result.info.status, RunStatus.RUNNING)

    @patch("trackio_mlflow.store.context_vars")
    @patch("trackio_mlflow.store.trackio")
    def test_update_run_info_with_end_status(
        self, mock_trackio: Mock, mock_context_vars: Mock
    ) -> None:
        """Test updating run info with an end status (FINISHED, FAILED, KILLED)."""
        run_id = "test_run_id"
        mock_run = Mock()
        mock_run.name = run_id
        self.store._run_map[run_id] = mock_run
        mock_context_vars.current_run.get.return_value = mock_run

        for end_status in [RunStatus.FINISHED, RunStatus.FAILED, RunStatus.KILLED]:
            with self.subTest(status=end_status):
                result = self.store.update_run_info(
                    run_id=run_id,
                    run_status=end_status,
                    end_time=9876543210,
                    run_name=self.run_name,
                )

                mock_trackio.finish.assert_called()
                self.assertIsInstance(result, RunInfo)
                self.assertEqual(result.run_id, run_id)
                self.assertEqual(result.status, end_status)
                self.assertEqual(result.end_time, 9876543210)

    def test_log_batch(self) -> None:
        """Test logging a batch of metrics and parameters."""
        run_id = "test_run_id"
        mock_run = Mock()
        self.store._run_map[run_id] = mock_run

        metrics = [
            Metric(key="accuracy", value=0.95, timestamp=123456, step=1),
            Metric(key="loss", value=0.05, timestamp=123457, step=1),
        ]
        params = [Param(key="learning_rate", value="0.01")]

        self.store.log_batch(
            run_id=run_id, metrics=metrics, params=params, tags=self.tags
        )

        expected_metrics = {"accuracy": 0.95, "loss": 0.05}
        mock_run.log.assert_called_once_with(metrics=expected_metrics, step=1)

    def test_get_run(self) -> None:
        """Test retrieving a run by ID."""
        run_id = "test_run_id"
        mock_run = Mock()
        mock_run.name = run_id
        self.store._run_map[run_id] = mock_run

        result = self.store.get_run(run_id)

        self.assertIsInstance(result, Run)
        self.assertEqual(result.info.run_id, run_id)
        self.assertEqual(result.info.run_uuid, run_id)
        self.assertEqual(result.info.status, RunStatus.RUNNING)
        self.assertEqual(result.info.lifecycle_stage, LifecycleStage.ACTIVE)
        self.assertEqual(result.info.artifact_uri, "file:///tmp/")
        self.assertIsInstance(result.data, RunData)

    def test_get_run_nonexistent(self) -> None:
        """Test retrieving a non-existent run raises KeyError."""
        with self.assertRaises(KeyError):
            self.store.get_run("nonexistent_run_id")
