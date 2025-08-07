# Trackio MLflow Plugin

> [!CAUTION]
> This project is a demonstration project for integrating Trackio with MLflow via the plugin. It is not intended for production use.

## Installation

Install via `pip`:

```bash
pip install git+https://github.com/rudeigerc/trackio-mlflow.git
```

## Usage

To use the Trackio MLflow plugin, you need to configure your MLflow tracking server to use Trackio as a tracking store. This can be done by setting the `MLFLOW_TRACKING_URI` environment variable to the Trackio server URL.

```bash
export MLFLOW_TRACKING_URI=trackio://
```

You can then run your MLflow experiments as usual, and the plugin will automatically log the metrics and parameters to Trackio.

## License

MIT
