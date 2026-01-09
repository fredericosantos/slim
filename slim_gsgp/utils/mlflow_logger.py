"""
MLflow logging utilities for SLIM GSGP algorithms.

  mlflow_logger.py
  ├── init_mlflow_run()        # Initializes MLflow tracking and starts a run
  ├── log_generation()         # Logs metrics for each generation
  ├── log_final_metrics()      # Logs final fitness values at end of run
  └── end_mlflow_run()         # Ends the active MLflow run
"""

import mlflow
from typing import Any


def init_mlflow_run(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any]
) -> None:
    """
    Initialize MLflow tracking and start a new run.

    Parameters
    ----------
    tracking_uri : str
        The MLflow tracking server URI (e.g., './mlruns' for local, or a remote server URL).
    experiment_name : str
        Name of the MLflow experiment.
    run_name : str
        Name for this specific run.
    params : dict[str, Any]
        Dictionary of parameters to log (e.g., seed, dataset, hyperparameters).
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(params)


def log_generation(
    generation: int,
    best_train_fitness: float,
    best_test_fitness: float | None,
    best_n_nodes: int,
    fitness_function: str
) -> None:
    """
    Log metrics for the current generation.

    Parameters
    ----------
    generation : int
        Current generation number.
    best_train_fitness : float
        Best training fitness value for this generation.
    best_test_fitness : float | None
        Best test fitness value for this generation (None if test_elite is False).
    best_n_nodes : int
        Number of nodes in the best individual's tree.
    fitness_function : str
        Name of the fitness function being used (e.g., 'rmse', 'mse', 'mae').
    """
    mlflow.log_metric("best_train_rmse", best_train_fitness, step=generation)
    if best_test_fitness is not None:
        mlflow.log_metric("best_test_rmse", best_test_fitness, step=generation)
    
    mlflow.log_metric("best_n_nodes", best_n_nodes, step=generation)


def log_final_metrics(
    elite,
    X_train,
    y_train,
    X_test,
    y_test,
    scaler,
    ffunction,
    operator: str = "sum"
) -> None:
    """
    Log final fitness values at the end of the run, evaluating on unscaled data.

    Parameters
    ----------
    elite : Individual
        The elite individual from the final generation.
    X_train : torch.Tensor
        Training input data.
    y_train : torch.Tensor
        Training output data (scaled).
    X_test : torch.Tensor | None
        Testing input data (None if not available).
    y_test : torch.Tensor | None
        Testing output data (scaled, None if not available).
    scaler : sklearn scaler object | None
        Scaler with inverse_transform method to unscale the data (None if no scaling used).
    ffunction : callable
        Fitness function to use for evaluation. Should accept (y_true, y_pred, scaler) if scaler is provided.
    operator : str, optional
        Operator to use for SLIM evaluation ('sum' or 'prod'). Default is 'sum'.
    """
    from slim_gsgp.utils.utils import _evaluate_slim_individual
    
    # Evaluate on training data with scaler
    if scaler is not None:
        train_fitness = _evaluate_slim_individual(
            elite, 
            lambda y_true, y_pred: ffunction(y_true, y_pred, scaler),
            y_train,
            testing=False,
            operator=operator
        )
    else:
        train_fitness = _evaluate_slim_individual(
            elite,
            ffunction,
            y_train,
            testing=False,
            operator=operator
        )
    
    mlflow.log_metric("best_train_fitness", float(train_fitness))
    
    # Evaluate on test data if available
    if X_test is not None and y_test is not None:
        # Calculate test semantics if not already done
        if elite.test_semantics is None:
            elite.calculate_semantics(X_test, testing=True)
        
        if scaler is not None:
            _evaluate_slim_individual(
                elite,
                lambda y_true, y_pred: ffunction(y_true, y_pred, scaler),
                y_test,
                testing=True,
                operator=operator
            )
        else:
            _evaluate_slim_individual(
                elite,
                ffunction,
                y_test,
                testing=True,
                operator=operator
            )
        
        mlflow.log_metric("best_test_fitness", float(elite.test_fitness))


def end_mlflow_run() -> None:
    """End the active MLflow run."""
    mlflow.end_run()
