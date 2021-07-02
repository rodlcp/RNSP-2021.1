import numpy as np
import wisardpkg as wp
from sklearn.metrics import accuracy_score

class DecodedModel():
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
    
    def fit(self, X, y):
        encoded_X = wp.DataSet(self.encoder.encode(X), y)
        self.model.train(encoded_X)
    
    def score(self, X, y):
        encoded_X = wp.DataSet(self.encoder.encode(X))
        y_predict = self.model.classify(encoded_X)
        return accuracy_score(y, y_predict)

class ThresholdEncoder():
    def __init__(self, threshold):
        self.threshold = threshold
    
    def encode(self, X):
        return X >= self.threshold

class ThermometerEncoder():
    def __init__(self, minimum, maximum, steps):
        self.minimum = minimum
        self.maximum = maximum
        self.steps = steps
        self.thresholds = np.linspace(minimum, maximum, steps, endpoint=False)
    
    def encode(self, X):
        X = np.asarray(X)
        shape = [self.steps] + list(X.shape)
        output = np.zeros(shape, dtype=np.bool)
        
        for i, j in enumerate(self.thresholds):
            output[i][X >= j] = 1

        output = np.transpose(output, (1, 2, 0)).reshape(X.shape[0], X.shape[1] * self.steps)
        return output
    
class PercentileThermometerEncoder():
    def __init__(self, steps, X):
        self.steps = steps
        self.percentiles = np.array([
            np.percentile(X, i, 0) for i in np.linspace(0, 100, steps, endpoint=False)
        ])
    
    def encode(self, X):
        X = np.asarray(X)
        shape = [self.steps] + list(X.shape)
        output = np.zeros(shape, dtype=np.bool)
        
        for i, j in enumerate(self.percentiles):
            output[i][X >= j] = 1

        output = np.transpose(output, (1, 2, 0)).reshape(X.shape[0], X.shape[1] * self.steps)
        return output

class AdaptativeThermometerEncoder():
    def __init__(self, X):
        self.minimum = X.min(0)
        self.maximum = X.max(0)
        self.std = X.std(0)
        self.steps = np.floor(self.std / self.std.min()).astype(int) + 1
    
    def encode(self, X):
        X = np.asarray(X)
        output = np.zeros((X.shape[0], self.steps.sum()), dtype=np.bool)
        
        for i, (m, M, s) in enumerate(zip(self.minimum, self.maximum, self.steps)):
            threshold = np.linspace(m, M, s, endpoint=False)
            prev_steps = self.steps[:i].sum()
            for j, t in enumerate(threshold):
                output[:, prev_steps + j] = X[:, i] >= t
        return output

class OneHotEncoder():
    def __init__(self, minimum, maximum, steps):
        self.minimum = minimum
        self.maximum = maximum
        self.steps = steps
        self.thresholds = np.linspace(minimum, maximum, steps + 1)
    
    def encode(self, X):
        X = np.asarray(X)
        shape = [self.steps] + list(X.shape)
        output = np.zeros(shape, dtype=np.bool)

        for i, (j, k) in enumerate(zip(self.thresholds[:-1], self.thresholds[1:])):
            output[i][(X >= j) & (X < k)] = 1

        output = np.transpose(output, (1, 2, 0)).reshape(X.shape[0], X.shape[1] * self.steps)
        return output

class PercentileOneHotEncoder():
    def __init__(self, steps, X):
        self.steps = steps
        self.percentiles = np.array([
            np.percentile(X, i, 0) for i in np.linspace(0, 100, steps + 1)
        ])
    
    def encode(self, X):
        X = np.asarray(X)
        shape = [self.steps] + list(X.shape)
        output = np.zeros(shape, dtype=np.bool)


        for i, (j, k) in enumerate(zip(self.percentiles[:-1], self.percentiles[1:])):
            output[i][(X >= j) & (X < k)] = 1

        output = np.transpose(output, (1, 2, 0)).reshape(X.shape[0], X.shape[1] * self.steps)
        return output