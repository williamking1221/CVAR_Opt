import warnings

import numpy as np
import pandas as pd


def _check_returns(returns):
    # Check NaNs excluding leading NaNs
    if np.any(np.isnan(returns.mask(returns.ffill().isnull(), 0))):
        warnings.warn(
            "Some returns are NaN. Please check your price data.", UserWarning
        )
    if np.any(np.isinf(returns)):
        warnings.warn(
            "Some returns are infinite. Please check your price data.", UserWarning
        )


def returns_from_prices(prices, log_returns=False):
    if log_returns:
        returns = np.log(1 + prices.pct_change()).dropna(how="all")
    else:
        returns = prices.pct_change().dropna(how="all")
    return returns


def mean_hist_returns(
        prices,
        returns_data=False,
        compounding=True,
        freq=252,
        log_returns=False
):
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("Prices not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)

    _check_returns(returns)
    if compounding:
        return (1 + returns).prod() ** (freq / returns.count()) - 1
    else:
        return returns.mean() * freq


def sim_returns_from_options_data(curr_price, options_df, dte, num_simulations, ticker):
    # Determine Implied Volatility (IV)
    atm_call_iv = options_df.at["ATM Call Option", "IV"] if "ATM Call Option" in options_df.index else None
    atm_put_iv = options_df.at["ATM Put Option", "IV"] if "ATM Put Option" in options_df.index else None

    if atm_call_iv is None and atm_put_iv is None:
        raise ValueError("No ATM Call or ATM Put Option available for implied volatility estimation")

    iv = atm_put_iv if atm_call_iv is None else (atm_call_iv if atm_put_iv is None else (atm_put_iv + atm_call_iv) / 2)

    # Calculate One-Day Standard Deviation of Log Returns
    one_day_stdev_log_returns = iv / np.sqrt((dte / 360) * 252)

    # Simulate Log Returns
    log_return_sim = np.random.normal(0, one_day_stdev_log_returns, num_simulations)

    # Convert Log Returns to Actual Returns
    underlying_return_sim = np.exp(log_return_sim) - 1

    options_dict = options_df.to_dict(orient='index')

    columns = [ticker]
    for key in options_dict:
        columns.append(key)

    # Initialize a list to store data for DataFrame
    sim_data = []

    # Iterate over each simulated underlying return
    for underlying_return in underlying_return_sim:
        option_returns = []
        underlying_sim_price_change = curr_price * underlying_return

        # Calculate options returns based on underlying return for the current simulation
        for option in options_dict.keys():
            option_price = options_dict[option]["Price"]
            option_delta = options_dict[option]["Delta"]
            option_gamma = options_dict[option]["Gamma"]
            option_theta = options_dict[option]["Theta"]

            # Taylor's approximation of BSM options
            if underlying_return > 0:
                if option.split()[1] == 'Call':
                    option_change = underlying_sim_price_change * option_delta + \
                            0.5 * option_gamma * (underlying_sim_price_change ** 2) + option_theta
                else:
                    option_change = underlying_sim_price_change * option_delta - \
                            0.5 * option_gamma * (underlying_sim_price_change ** 2) + option_theta
            else:
                if option.split()[1] == 'Call':
                    option_change = underlying_sim_price_change * option_delta - \
                            0.5 * option_gamma * (underlying_sim_price_change ** 2) + option_theta
                else:
                    option_change = underlying_sim_price_change * option_delta + \
                            0.5 * option_gamma * (underlying_sim_price_change ** 2) + option_theta

            option_pct_change = option_change / option_price
            option_returns.append(option_pct_change)

        # Append the SPY stock return and options returns for the current simulation to the data list
        sim_data.append([underlying_return] + option_returns)

    sim_returns = pd.DataFrame(sim_data, columns=columns)
    mean_returns = sim_returns.mean()
    stdev_returns = sim_returns.std()
    return sim_returns, mean_returns, stdev_returns

