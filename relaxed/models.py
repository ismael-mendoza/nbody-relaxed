import warnings
from abc import ABC
from abc import abstractmethod

import numpy as np
from scipy.interpolate import interp1d
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import QuantileTransformer

from relaxed.analysis import get_an_from_am


class PredictionModel(ABC):
    def __init__(self, n_features: int, n_targets: int) -> None:
        assert isinstance(n_features, int) and n_features > 0
        self.n_features = n_features
        self.n_targets = n_targets
        self.trained = False  # whether model has been trained yet.

    def fit(self, x, y):
        # change internal state such that predict and/or sampling are possible.
        assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)) == 0
        assert x.shape == (y.shape[0], self.n_features)
        y = y.reshape(x.shape[0], self.n_targets)
        self._fit(x, y)
        self.trained = True

    def predict(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained
        return self._predict(x).reshape(x.shape[0], self.n_targets)

    @abstractmethod
    def _fit(self, x, y):
        pass

    @abstractmethod
    def _predict(self, x):
        pass


class PredictionModelTransform(PredictionModel, ABC):
    """Enable possibility of transforming variables at fitting/prediction time."""

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        to_log: bool = False,
        to_marginal_normal: bool = True,
        use_multicam: bool = True,
    ) -> None:
        super().__init__(n_features, n_targets)
        assert (to_log + to_marginal_normal) <= 1, "Only 1 transformation at a time."

        self.to_marginal_normal = to_marginal_normal  # transform data to be (marginally) gaussian
        self.to_log = to_log  # transform to log space.
        self.use_multicam = use_multicam  # map predictions to trained data quantiles.

        # quantile transformers.
        self.qt_x = None
        self.qt_y = None

    def fit(self, x, y):
        y = y.reshape(x.shape[0], self.n_targets)

        if self.use_multicam or self.to_marginal_normal:
            self.qt_y = QuantileTransformer(n_quantiles=len(y), output_distribution="normal")
            self.qt_y = self.qt_y.fit(y)

        if self.to_marginal_normal:
            self.qt_x = QuantileTransformer(n_quantiles=len(x), output_distribution="normal")
            self.qt_x = self.qt_x.fit(x)

            # transform
            x_trans = self.qt_x.transform(x)
            y_trans = self.qt_y.transform(y)

            super().fit(x_trans, y_trans)

        elif self.to_log:
            super().fit(np.log(x), np.log(y))

        else:
            super().fit(x, y)

    def predict(self, x):
        if self.to_marginal_normal:
            x_trans = self.qt_x.transform(x)
            y_trans = super().predict(x_trans)
            y_pred = self.qt_y.inverse_transform(y_trans)

        elif self.to_log:
            y_pred = np.exp(super().predict(np.log(x)))

        else:
            y_pred = super().predict(x)

        # optionally map prediction to correct quantiles from trained distribution.
        if self.use_multicam:
            qt_pred = QuantileTransformer(n_quantiles=len(y_pred), output_distribution="normal")
            qt_pred = qt_pred.fit(y_pred)
            y_pred = self.qt_y.inverse_transform(qt_pred.transform(y_pred))

        return y_pred


class SamplingModel(PredictionModelTransform):
    def sample(self, x, n_samples):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained

        n_points = x.shape[0]
        if self.to_marginal_normal:
            x_trans = self.qt_x.transform(x)
        elif self.to_log:
            x_trans = np.log(x)
        else:
            x_trans = x

        y_pred = self._sample(x_trans, n_samples).reshape(n_points, n_samples, self.n_targets)

        if self.to_marginal_normal:
            for i in range(n_samples):
                y_pred[:, i, :] = self.qt_y.inverse_transform(y_pred[:, i, :])

        elif self.to_log:
            y_pred = np.exp(y_pred)

        if self.use_multicam:
            for i in range(n_samples):
                y_pred_i = y_pred[:, i, :]
                qt_pred = QuantileTransformer(
                    n_quantiles=len(y_pred_i), output_distribution="normal"
                )
                y_pred[:, i, :] = self.qt_y.inverse_transform(qt_pred.transform(y_pred_i))

        return y_pred

    @abstractmethod
    def _sample(self, x, n_samples):
        pass


class LogNormalRandomSample(PredictionModel):
    """Lognormal random samples."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        super().__init__(n_features, n_targets)

        self.mu = None
        self.sigma = None

    def _fit(self, x, y):
        assert np.all(y > 0)
        mu, sigma = np.mean(np.log(y), axis=0), np.std(np.log(y), axis=0)
        self.mu = mu
        self.sigma = sigma

    def _predict(self, x):
        n_test = len(x)
        return np.exp(np.random.normal(self.mu, self.sigma, (n_test, self.n_targets)))


class InverseCDFRandomSamples(PredictionModel):
    """Use Quantile Transformer to get random samples from a 1D Distribution."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        super().__init__(n_features, n_targets)
        assert self.n_targets == 1

    def _fit(self, x, y):
        self.qt_y = QuantileTransformer(n_quantiles=len(y), output_distribution="uniform")
        self.qt_y = self.qt_y.fit(y)

    def _predict(self, x):
        u = np.random.random(size=(len(x), self.n_targets))
        return self.qt_y.inverse_transform(u)


class LinearRegression(PredictionModelTransform):
    def __init__(self, n_features: int, n_targets: int, **transform_kwargs) -> None:
        super().__init__(n_features, n_targets, **transform_kwargs)
        self.reg = None

    def _fit(self, x, y):
        self.reg = linear_model.LinearRegression().fit(x, y)

    def _predict(self, x):
        return self.reg.predict(x)


class LASSO(PredictionModelTransform):
    name = "lasso"

    def __init__(
        self, n_features: int, n_targets: int, alpha: float = 0.1, **transform_kwargs
    ) -> None:
        # alpha is the regularization parameter.
        super().__init__(n_features, n_targets, **transform_kwargs)
        self.alpha = alpha

        # attributes of fit
        self.lasso = None
        self.importance = None

    def _fit(self, x, y):
        # use lasso linear regression.
        _lasso = linear_model.Lasso(alpha=self.alpha)
        selector = SelectFromModel(estimator=_lasso).fit(x, y)
        self.lasso = _lasso.fit(x, y)
        self.importance = selector.estimator.coef_

    def _predict(self, x):
        return self.lasso.predict(x)


class MultiVariateGaussian(SamplingModel):
    """Multi-Variate Gaussian using full covariance matrix (returns conditional mean)."""

    def __init__(self, n_features: int, n_targets, **transform_kwargs) -> None:
        # do_sample: wheter to sample from x1|x2 or return E[x1 | x2] for predictions
        super().__init__(n_features, n_targets, **transform_kwargs)

        self.mu1 = None
        self.mu2 = None
        self.Sigma = None
        self.rho = None
        self.sigma_cond = None

        if not self.to_marginal_normal:
            warnings.warn(
                "You initialized the `MultivariateGaussian` model with `to_marginal_normal`"
                "set to False. However, this model usually requires this data preprocessing step."
            )

    def _fit(self, x, y):
        """
        We assume a multivariate-gaussian distribution P(X, a(m1), a(m2), ...) with
        conditional distribution P(X | {a(m_i)}) uses the rule here:
        https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
        we return the mean/std deviation of the conditional gaussian.

        * y (usually) represents one of the dark matter halo properties at z=0.
        * x are the features used for prediction, should have shape (y.shape[0], n_features)
        """
        n_features = self.n_features
        n_targets = self.n_targets

        # calculate sigma/correlation matrix bewteen all quantities
        z = np.hstack([y.reshape(-1, n_targets), x])

        # some sanity checks
        assert z.shape == (y.shape[0], n_targets + n_features)
        np.testing.assert_equal(y, z[:, :n_targets])
        np.testing.assert_equal(x[:, 0], z[:, n_targets])  # ignore mutual nan's
        np.testing.assert_equal(x[:, -1], z[:, -1])

        # calculate covariances
        total_features = n_targets + n_features
        Sigma = np.zeros((total_features, total_features))
        rho = np.zeros((total_features, total_features))
        for i in range(total_features):
            for j in range(total_features):
                if i <= j:
                    # calculate correlation coefficient keeping only non-nan values
                    z1, z2 = z[:, i], z[:, j]
                    keep = ~np.isnan(z1) & ~np.isnan(z2)
                    cov = np.cov(z1[keep], z2[keep])
                    assert cov.shape == (2, 2)
                    Sigma[i, j] = cov[0, 1]
                    rho[i, j] = np.corrcoef(z1[keep], z2[keep])[0, 1]
                else:
                    rho[i, j] = rho[j, i]
                    Sigma[i, j] = Sigma[j, i]

        # more sanity checks.
        assert np.all(~np.isnan(Sigma))
        assert np.all(~np.isnan(rho))

        mu1 = np.nanmean(y, axis=0).reshape(n_targets, 1)
        mu2 = np.nanmean(x, axis=0).reshape(n_features, 1)
        Sigma11 = Sigma[:n_targets, :n_targets].reshape(n_targets, n_targets)
        Sigma12 = Sigma[:n_targets, n_targets:].reshape(n_targets, n_features)
        Sigma22 = Sigma[n_targets:, n_targets:].reshape(n_features, n_features)
        sigma_bar = Sigma11 - Sigma12.dot(np.linalg.inv(Sigma22)).dot(Sigma12.T)

        # update prediction attributes
        self.mu1 = mu1
        self.mu2 = mu2
        self.Sigma = Sigma
        self.Sigma11 = Sigma11
        self.Sigma12 = Sigma12
        self.Sigma22 = Sigma22
        self.rho = rho
        self.sigma_bar = sigma_bar.reshape(n_targets, n_targets)

    def _get_mu_cond(self, x):
        # returns mu_cond evaluated at given x.
        assert self.trained
        assert np.sum(np.isnan(x)) == 0
        n_points = x.shape[0]
        x = x.reshape(n_points, self.n_features).T
        mu_cond = self.mu1 + self.Sigma12.dot(np.linalg.inv(self.Sigma22)).dot(x - self.mu2)
        return mu_cond.T.reshape(n_points, self.n_targets)

    def _predict(self, x):
        return self._get_mu_cond(x)

    def _sample(self, x, n_samples):
        n_points = x.shape[0]
        _zero = np.zeros((n_samples, self.n_targets))
        mu_cond = self._get_mu_cond(x)
        size = (n_points, n_samples)
        y_pred = np.random.multivariate_normal(mean=_zero, cov=self.sigma_bar, size=size)
        assert y_pred.shape == (n_points, n_samples, self.n_targets)
        y_pred += mu_cond.reshape(-1, 1, self.n_targets)
        return y_pred


class CAM(PredictionModel):
    """Conditional Abundance Matching"""

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        mass_bins: np.ndarray,
        mbin: float,
        cam_order: int = -1,
    ) -> None:
        # cam_order: +1 or -1 depending on correlation of a_{n} with y
        assert n_features == len(mass_bins)
        assert n_targets == 1
        super().__init__(n_features, n_targets)

        assert cam_order in {-1, 1}
        assert isinstance(mass_bins, np.ndarray)
        self.mbin = mbin
        self.cam_order = cam_order
        self.mass_bins = mass_bins

        # fit attributes
        self.an_to_mark = None
        self.mark_to_Y = None

    def _fit(self, am, y):

        y = y.reshape(-1)
        an_train = get_an_from_am(am, self.mass_bins, mbin=self.mbin).reshape(-1)
        assert an_train.shape[0] == am.shape[0]

        y_sort, an_sort = self.cam_order * np.sort(self.cam_order * y), np.sort(an_train)
        marks = np.arange(len(y_sort)) / len(y_sort)
        marks += (marks[1] - marks[0]) / 2
        self.an_to_mark = interp1d(an_sort, marks, fill_value=(0, 1), bounds_error=False)
        self.mark_to_Y = interp1d(
            marks, y_sort, fill_value=(y_sort[0], y_sort[-1]), bounds_error=False
        )

    def _predict(self, am):
        an = get_an_from_am(am, self.mass_bins, mbin=self.mbin)
        return self.mark_to_Y(self.an_to_mark(an))


class BayesianLinearRegression(SamplingModel):
    def __init__(self, n_features, n_targets, mu0, sigma0, noise_var, **kwargs) -> None:
        super().__init__(n_features, n_targets, **kwargs)

        self.prior = (mu0, sigma0)

        # prior on beta
        # assumes data is already normal gaussian transformed.
        self.mu = mu0
        self.sigma = sigma0

        # model
        self.noise_var = noise_var  # assumes diagonal, uniform covariance.

        assert self.mu.shape[0] == self.n_features
        assert self.sigma.shape == (self.n_features, self.n_features)

    def _gaussian_inverse_problem(self, x, y, mu0, sigma0, noise_var):
        # Y is data
        # 0 mean prior
        # Sigma0 is prior on Beta covariance.
        # A is vandermonde matrix
        n_points, _ = x.shape
        assert n_points == y.shape[0]
        S = x.dot(sigma0).dot(x.T) + noise_var * np.eye(n_points)
        U = sigma0.dot(x.T)

        delta = y - x.dot(mu0)
        mu_post = U.dot(np.linalg.solve(S, delta))
        sigma_post = sigma0 - U.dot(np.linalg.inv(S)).dot(U.T)

        return mu_post, sigma_post

    def _fit(self, x, y):
        self.mu, self.sigma = self._gaussian_inverse_problem(
            x, y, self.mu, self.sigma, self.noise_var
        )

    def _predict(self, x):
        # return expectation value / MAP.
        n_points = x.shape[0]
        return x.dot(self.mu).reshape(n_points, self.n_targets)

    def _sample(self, x, n_samples):
        n_points = x.shape[0]
        y_samples = np.zeros((n_points, n_samples, self.n_targets))
        for i in range(n_points):
            xi = x[i, None]
            mean = xi.dot(self.mu).reshape(-1)
            cov = xi.dot(self.sigma).dot(xi.T) + self.noise_var * np.eye(1)
            y_samples[i] = np.random.multivariate_normal(mean, cov, size=(n_samples,))
        return y_samples


available_models = {
    "gaussian": MultiVariateGaussian,
    "cam": CAM,
    "linear": LinearRegression,
    "lasso": LASSO,
    "lognormal": LogNormalRandomSample,
    "bayes_linear": BayesianLinearRegression,
}


def training_suite(data: dict):
    """Returned models specified in the data dictionary.

    Args:
        data:  Dictionary containing all the information required to train models. Using the format
            `name:info` where `name` is an identifier for the model (can be anything)
            and `info` is a dictionary with keys:
                - 'xy': (x,y) tuple containing data to train model with.
                - 'model': Which model from `available_models` to use.
                - 'n_features': Number of features for this model.
                - 'kwargs': Keyword argument dict to initialize the model.
    """
    # check data dict is in the right format.
    assert isinstance(data, dict)
    for name in data:
        assert isinstance(data[name]["xy"], tuple)
        assert data[name]["model"] in available_models
        assert isinstance(data[name]["n_features"], int)
        assert isinstance(data[name]["n_targets"], int)
        assert isinstance(data[name]["kwargs"], dict)
        assert data[name]["n_features"] == data[name]["xy"][0].shape[1]

    trained_models = {}
    for name in data:
        m = data[name]["model"]
        kwargs = data[name]["kwargs"]
        n_features = data[name]["n_features"]
        n_targets = data[name]["n_targets"]
        x, y = data[name]["xy"]
        model = available_models[m](n_features, n_targets, **kwargs)
        model.fit(x, y)
        trained_models[name] = model

    return trained_models
