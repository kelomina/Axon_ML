import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class ProbabilityCalibrator:
    def __init__(self, method="isotonic"):
        self.method = method
        self.model = None

    def fit(self, y_true, y_prob):
        y_true = np.asarray(y_true).astype(np.int64)
        y_prob = np.asarray(y_prob).astype(np.float32)
        if not (0 < y_prob.min() < y_prob.max() < 1):
            y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(y_prob, y_true)
        else:
            logit = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(logit, y_true)
            self.model = lr
        return self

    def predict(self, y_prob):
        y_prob = np.asarray(y_prob).astype(np.float32)
        if not (0 < y_prob.min() < y_prob.max() < 1):
            y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
        if isinstance(self.model, IsotonicRegression):
            return self.model.predict(y_prob)
        logit = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
        return self.model.predict_proba(logit)[:, 1]
