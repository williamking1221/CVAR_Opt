
import warnings
import collections

import cvxpy as cp
import numpy as np
import pandas as pd

from . import exceptions


class CVaROpt:
    def __init__(self,
                 mean_returns,
                 sim_returns,
                 num_samples,
                 beta=0.95,
                 percent_bounds=True,
                 weight_bounds=(-1, 1),
                 money_bounds=None,
                 money_limit=10000,
                 solver=None,
                 verbose=False,
                 solver_options=None):
        if mean_returns is not None:
            self.num_assets = len(mean_returns)
            self.mean_returns = mean_returns
        else:
            raise Exception("Lacking mean_returns value")

        if isinstance(mean_returns, pd.Series):
            tickers = list(mean_returns.index)
        else:
            tickers = list(range(self.num_assets))

        if tickers is None:
            self.tickers = list(range(self.num_assets))
        else:
            self.tickers = tickers
        self._risk_free_rate = None
        self.weights = None
        self.monetary_inv = None

        self._w = cp.Variable(self.num_assets)
        self._m = cp.Variable(self.num_assets)
        self._obj = None
        self._additional_obj = []
        self._constraints = []
        self.percent_bounds = percent_bounds
        self._lower_bounds_weight = None
        self._upper_bounds_weight = None
        self._lower_bounds_money = None
        self._upper_bounds_money = None
        self._opt = None
        self._solver = solver
        self._verbose = verbose
        self._solver_options = solver_options if solver_options else {}

        if percent_bounds:
            self._map_weight_bounds_to_constraints = self._map_weight_bounds_to_constraints(weight_bounds)
        else:
            self._map_money_bounds_to_constraints = self._map_money_bounds_to_constraints(money_bounds, money_limit)

        self.num_samples = num_samples
        self.money_limit = money_limit

        self._validate_mean_returns(mean_returns)
        self._req_return = None
        self._target_cvar = None
        self._market_neutral = None

        self.sim_returns = self._validate_sim_returns(sim_returns)
        self._beta = self._check_beta(beta)
        self._alpha = cp.Variable()
        self._u = cp.Variable(len(self.sim_returns))

    @staticmethod
    def _validate_mean_returns(expected_returns):
        if expected_returns is None:
            return None
        elif isinstance(expected_returns, pd.Series):
            return expected_returns.values
        elif isinstance(expected_returns, list):
            return np.array(expected_returns)
        elif isinstance(expected_returns, np.ndarray):
            return expected_returns.ravel()
        else:
            raise TypeError("expected_returns is not a series, list or array")

    @staticmethod
    def _check_beta(beta):
        if not (0 <= beta < 1):
            raise ValueError("Beta must be between 0 and 1")
        if beta <= 0.25:
            warnings.warn(
                "Warning: Beta is the confidence level, not quantile. Typical vals are 0.80, 0.9, 0.95, 0.99",
                UserWarning,
            )
        return beta

    def _validate_sim_returns(self, sim_returns):
        """
        Helper method to validate daily returns (needed for some efficient frontiers)
        """
        if not isinstance(sim_returns, (pd.DataFrame, np.ndarray)):
            raise TypeError("returns should be a pd.Dataframe or np.ndarray")

        returns_df = pd.DataFrame(sim_returns)
        if returns_df.isnull().values.any():
            warnings.warn(
                "Removing NaNs from returns",
                UserWarning,
            )
            returns_df = returns_df.dropna(axis=0, how="any")

        if self.mean_returns is not None:
            if returns_df.shape[1] != len(self.mean_returns):
                raise ValueError(
                    "returns columns do not match expected_returns. Please check your tickers."
                )

        return returns_df

    def _map_weight_bounds_to_constraints(self, test_bounds):
        # If it is a collection with the right length, assume they are all bounds.
        if len(test_bounds) == self.num_assets and not isinstance(
            test_bounds[0], (float, int)
        ):
            bounds = np.array(test_bounds, dtype=float)
            self._lower_bounds_weight = np.nan_to_num(bounds[:, 0], nan=-np.inf)
            self._upper_bounds_weight = np.nan_to_num(bounds[:, 1], nan=np.inf)
        else:
            # Otherwise this must be a pair.
            if len(test_bounds) != 2 or not isinstance(test_bounds, (tuple, list)):
                raise TypeError(
                    "test_bounds must be a pair (lower bound, upper bound) OR a collection of bounds for each asset"
                )
            lower, upper = test_bounds

            # Replace None values with the appropriate +/- 1
            if np.isscalar(lower) or lower is None:
                lower = -1 if lower is None else lower
                self._lower_bounds_weight = np.array([lower] * self.num_assets)
                upper = 1 if upper is None else upper
                self._upper_bounds_weight = np.array([upper] * self.num_assets)
            else:
                self._lower_bounds_weight = np.nan_to_num(lower, nan=-1)
                self._upper_bounds_weight = np.nan_to_num(upper, nan=1)

        self.add_constraint(lambda w: w >= self._lower_bounds_weight)
        self.add_constraint(lambda w: w <= self._upper_bounds_weight)
        self.add_constraint(lambda w: cp.sum(w) == 1)

    def _map_money_bounds_to_constraints(self, test_bounds, test_limit, set_limit=100000):
        # If it is a collection with the right length, assume they are all bounds.
        if test_bounds is not None and len(test_bounds) == self.num_assets and not isinstance(
            test_bounds[0], (float, int)
        ):
            bounds = np.array(test_bounds, dtype=float)
            self._lower_bounds_money = np.nan_to_num(bounds[:, 0], nan=-set_limit)
            self._upper_bounds_money = np.nan_to_num(bounds[:, 1], nan=set_limit)
        else:
            # Otherwise this must be a pair.
            if test_bounds is not None and (len(test_bounds) != 2 or not isinstance(test_bounds, (tuple, list))):
                raise TypeError(
                    "test_bounds must be a pair (lower bound, upper bound) OR a collection of bounds for each asset"
                )
            if test_bounds is not None:
                lower, upper = test_bounds
            else:
                lower = None
                upper = None

            # Replace None values with the appropriate +/- 1
            if np.isscalar(lower) or lower is None:
                lower = set_limit if lower is None else lower
                self._lower_bounds_money = np.array([lower] * self.num_assets)
                upper = set_limit if upper is None else upper
                self._upper_bounds_money = np.array([upper] * self.num_assets)
            else:
                self._lower_bounds_money = np.nan_to_num(lower, nan=set_limit)
                self._upper_bounds_money = np.nan_to_num(upper, nan=set_limit)

        self.add_constraint(lambda m: m >= self._lower_bounds_money)
        self.add_constraint(lambda m: m <= self._upper_bounds_money)
        self.add_constraint(lambda m: cp.sum(m) == test_limit)

    def add_constraint(self, new_constraint):
        if not callable(new_constraint):
            raise TypeError(
                "New constraint must be callable (ex. Lambda function)"
            )
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "Adding constraints to already solved problem may have consequences. "
                "Create a new instance for the new set of constraints"
            )
        self._constraints.append(new_constraint(self._w))

    def _make_output_weights(self, weights=None):
        if weights is None:
            weights = self.weights

        return collections.OrderedDict(zip(self.tickers, weights))

    def _make_output_investment(self, investment=None):
        if investment is None:
            investment = self.monetary_inv

        return collections.OrderedDict(zip(self.tickers, investment))

    def _solve_cvxpy_opt_problem(self):
        try:
            if self._opt is None:
                self._opt = cp.Problem(cp.Minimize(self._obj), self._constraints)
                self._initial_objective = self._obj.id
                self._initial_constraint_ids = {const.id for const in self._constraints}
            else:
                if not self._obj.id == self._initial_objective:
                    raise exceptions.InstantiationError(
                        "The objective function was changed after the initial optimization. "
                        "Please create a new instance instead."
                    )

                constr_ids = {const.id for const in self._constraints}
                if not constr_ids == self._initial_constraint_ids:
                    raise exceptions.InstantiationError(
                        "The constraints were changed after the initial optimization. "
                        "Please create a new instance instead."
                    )
            self._opt.solve(
                solver=self._solver, verbose=self._verbose, **self._solver_options
            )

        except (TypeError, cp.DCPError) as e:
            raise exceptions.OptimizationError from e

        if self._opt.status not in {"optimal", "optimal_inaccurate"}:
            raise exceptions.OptimizationError(
                "Solver status: {}".format(self._opt.status)
            )
        if self.percent_bounds:
            self.weights = self._w.value.round(16) + 0.0  # +0.0 removes signed zero
            return self._make_output_weights()
        else:
            self.monetary_inv = self._m.value.round(3) + 0.0 # +0.0 removes signed zero
            return self._make_output_investment()

    def min_cvar(self):
        self._obj = self._alpha + 1.0 / (self.num_samples * (1 - self._beta)) * cp.sum(self._u)

        for obj in self._additional_obj:
            self._obj += obj

        self.add_constraint(lambda _: self._u >= 0.0)
        self.add_constraint(lambda w: self.sim_returns.values @ w + self._alpha + self._u >= 0.0)

        return self._solve_cvxpy_opt_problem()

    def min_cvar_with_req_returns(self, req_return):
        self._req_return = req_return

        self._obj = self._alpha + 1.0 / (self.num_samples * (1 - self._beta)) * cp.sum(self._u)

        for obj in self._additional_obj:
            self._obj += obj

        self.add_constraint(lambda _: self._u >= 0.0)
        self.add_constraint(lambda w: self.sim_returns.values @ w + self._alpha + self._u >= 0.0)

        ret = self.mean_returns.values @ self._w
        req_return_par = cp.Parameter(name="req_return", value=req_return)
        self.add_constraint(lambda _: ret >= req_return_par)

        return self._solve_cvxpy_opt_problem()

    def max_return_with_req_cvar(self, target_cvar):
        self._target_cvar = target_cvar
        self._obj = portfolio_return(self._w, self.mean_returns)
        for obj in self._additional_obj:
            self._obj += obj

        cvar = self._alpha + 1.0 / (self.num_samples * (1 - self._beta)) * cp.sum(self._u)
        target_cvar_par = cp.Parameter(value=target_cvar, name="target_cvar")

        self.add_constraint(lambda _: cvar <= target_cvar_par)
        self.add_constraint(lambda _: self._u >= 0.0)
        self.add_constraint(lambda w: self.sim_returns.values @ w + self._alpha + self._u >= 0.0)

        return self._solve_cvxpy_opt_problem()

    def portfolio_performance(self, verbose=False):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio, specifically: expected return, CVaR

        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :raises ValueError: if weights have not been calculated yet
        :return: expected return, CVaR.
        :rtype: (float, float)
        """
        mu = portfolio_return(
            self.weights, self.mean_returns, negative=False
        )

        cvar = self._alpha + 1.0 / (self.num_samples * (1 - self._beta)) * cp.sum(self._u)
        cvar_val = cvar.value

        if verbose:
            print("Expected daily return: {:.1f}%".format(100 * mu))
            print("Conditional Value at Risk: {:.2f}%".format(100 * cvar_val))

        return mu, cvar_val


def _objective_value(w, obj):
    """
    Helper method to return either the value of the objective function
    or the objective function as a cvxpy object depending on whether
    w is a cvxpy variable or np array.

    :param w: weights
    :type w: np.ndarray OR cp.Variable
    :param obj: objective function expression
    :type obj: cp.Expression
    :return: value of the objective function OR objective function expression
    :rtype: float OR cp.Expression
    """
    if isinstance(w, np.ndarray):
        if np.isscalar(obj):
            return obj
        elif np.isscalar(obj.value):
            return obj.value
        else:
            return obj.value.item()
    else:
        return obj


def portfolio_return(w, expected_returns, negative=True):
    """
    Calculate the (negative) mean return of a portfolio

    :param w: asset weights in the portfolio
    :type w: np.ndarray OR cp.Variable
    :param expected_returns: expected return of each asset
    :type expected_returns: np.ndarray
    :param negative: whether quantity should be made negative (so we can minimise)
    :type negative: boolean
    :return: negative mean return
    :rtype: float
    """
    sign = -1 if negative else 1
    mu = w @ expected_returns
    return _objective_value(w, sign * mu)
