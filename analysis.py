"""Time series analysis module"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from data import t as t_orig, y as y_orig


def seasonal_func(x, a, b, w, phi, c):
    """Seasonal function: a + b * sin(w * x + phi) + c * x"""
    return a + b * np.sin(w * x + phi) + c * x


def fit_linear_trend(t, y):
    """Fit linear trend using machine learning."""
    t_reshaped = t.reshape(-1, 1)
    linreg = LinearRegression()
    linreg.fit(t_reshaped, y)
    trend = linreg.predict(t_reshaped)
    return linreg, trend


def fit_seasonal_trend(t, y):
    """
    Fit a seasonal trend to the time series data using linear regression.
    The model uses sine and cosine features to capture seasonality, and a linear term for trend.
    Args:
        t (array-like): Time points (e.g., months).
        y (array-like): Observed values (e.g., revenue).
    Returns:
        reg: Fitted LinearRegression model.
        omega: Angular frequency for the seasonal component.
    """
    period = 12  # Assume seasonality repeats every 12 time units (e.g., months)
    omega = 2 * np.pi / period  # Calculate angular frequency for the seasonal component
    X = np.column_stack(
        [
            np.ones_like(t),  # Intercept
            np.sin(omega * t),  # Sine seasonal feature
            np.cos(omega * t),  # Cosine seasonal feature
            t,  # Linear trend feature
        ]
    )
    reg = LinearRegression()  # Create linear regression model
    reg.fit(X, y)  # Fit model to data
    return reg, omega  # Return fitted model and frequency


def forecast(linreg, seasonal_reg, omega, t_future):
    """Forecast future values using both trends."""
    y_pred_linear = linreg.predict(np.array([[t_future]]))[0]
    X_future = np.column_stack(
        [np.ones(1), np.sin(omega * t_future), np.cos(omega * t_future), [t_future]]
    )
    y_pred_seasonal = seasonal_reg.predict(X_future)[0]
    return y_pred_linear, y_pred_seasonal


def plot_results(t, y, trend, t_ext, seasonal_trend_ext):
    """Plot the original data and the fitted trends."""
    axs = plt.subplots(1, 2, figsize=(12, 5))[1]
    # First plot: original data
    axs[0].plot(t, y, "o-", label="Actual data")
    axs[0].set_xlabel("Month")
    axs[0].set_ylabel("Revenue, mln RUB")
    axs[0].set_title("Revenue time series")
    axs[0].legend()
    # Second plot: trends
    axs[1].plot(t, y, "o", label="Actual data")
    axs[1].plot(t, trend, "-", label="Linear trend (ML)")
    axs[1].plot(t_ext, seasonal_trend_ext, "--", label="Seasonal trend (ML)")
    axs[1].set_xlabel("Month")
    axs[1].set_ylabel("Revenue, mln RUB")
    axs[1].set_title("Trends: linear and seasonal")
    axs[1].legend()
    plt.tight_layout()
    plt.show()


def run_analysis():
    """Run time series analysis on the provided data."""
    t_future = 11

    t = np.array(t_orig)
    y = np.array(y_orig)

    linreg, trend = fit_linear_trend(t, y)
    seasonal_reg, omega = fit_seasonal_trend(t, y)

    y_pred_linear, y_pred_seasonal = forecast(linreg, seasonal_reg, omega, t_future)
    t_ext = np.append(t, t_future)
    y_ext = np.append(y, y_pred_seasonal)

    # Refit seasonal model with extended data
    seasonal_reg_ext, omega_ext = fit_seasonal_trend(t_ext, y_ext)
    X_ext = np.column_stack(
        [
            np.ones_like(t_ext),
            np.sin(omega_ext * t_ext),
            np.cos(omega_ext * t_ext),
            t_ext,
        ]
    )
    seasonal_trend_ext = seasonal_reg_ext.predict(X_ext)

    plot_results(t, y, trend, t_ext, seasonal_trend_ext)

    print(
        f"Revenue forecast for month 11 (linear trend, ML): {y_pred_linear:.2f} mln RUB"
    )
    print(
        f"Revenue forecast for month 11 (seasonal trend, ML): {y_pred_seasonal:.2f} mln RUB"
    )
