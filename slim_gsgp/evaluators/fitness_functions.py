# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module provides various error metrics functions for evaluating machine learning models.
"""

import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        RMSE value.
    """
    return torch.sqrt(torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1))


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MSE value.
    """
    return torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MAE value.
    """
    return torch.mean(torch.abs(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)


def mae_int(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE) for integer values.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MAE value for integer predictions.
    """
    return torch.mean(torch.abs(torch.sub(y_true, torch.round(y_pred))), dim=len(y_pred.shape) - 1)


def signed_errors(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute signed errors between true and predicted values.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        Signed error values.
    """
    return torch.sub(y_true, y_pred)

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute R-squared (R²) score.

    If using this fitness function, please ensure that you are maximizing the fitness value when

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        R² score value.
    """
    ss_res = torch.sum(torch.square(y_true - y_pred))
    ss_tot = torch.sum(torch.square(y_true - torch.mean(y_true)))
    r2 = 1 - (ss_res / ss_tot)
    return r2


def rmse_with_scaler(y_true: torch.Tensor, y_pred: torch.Tensor, scaler) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE) after unscaling predictions.
    
    This function unscales both y_true and y_pred using the provided scaler
    before computing RMSE. The scaler is expected to be a scikit-learn scaler
    (e.g., StandardScaler, MinMaxScaler) with inverse_transform method.
    
    Parameters
    ----------
    y_true : torch.Tensor
        True values (scaled).
    y_pred : torch.Tensor
        Predicted values (scaled).
    scaler : sklearn scaler object
        Scaler with inverse_transform method to unscale the data.
    
    Returns
    -------
    torch.Tensor
        RMSE value computed on unscaled data.
    """
    import numpy as np
    
    # Convert to numpy, reshape for scaler, and unscale
    y_true_np = y_true.detach().cpu().numpy().reshape(-1, 1)
    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1, 1)
    
    y_true_unscaled = scaler.inverse_transform(y_true_np).flatten()
    y_pred_unscaled = scaler.inverse_transform(y_pred_np).flatten()
    
    # Convert back to torch tensors
    y_true_torch = torch.from_numpy(y_true_unscaled)
    y_pred_torch = torch.from_numpy(y_pred_unscaled)
    
    # Compute RMSE on unscaled data
    return torch.sqrt(torch.mean(torch.square(torch.sub(y_true_torch, y_pred_torch))))
