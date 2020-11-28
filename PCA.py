import numpy as np


class PCAEigenFace:
    def __init__(self, num_components=10):
        self.numComponents = num_components
        self.mean = None
        self.diffs = None
        self.covariance = None
        self.eigenValues = None
        self.eigenVectors = None
        self.eigenFaces = None
        self.labels = None
        self.projections = None

    def train(self, persons):
        self.calcMean(persons)
        self.calcDiff(persons)
        self.calcCovariance()
        self.calcEigen()
        self.calcEigenFaces()
        self.calcProjections(persons)

    def calcMean(self, persons):
        self.mean = np.mean(np.asarray(list(map(lambda p: p.data, persons))), axis=0)

    def calcDiff(self, persons):
        self.diffs = []
        for person in persons:
            self.diffs.append(person.data - self.mean)
        self.diffs = np.asarray(self.diffs).transpose()

    def calcCovariance(self):
        self.covariance = np.dot(self.diffs.transpose(), self.diffs)

    def calcEigen(self):
        self.eigenValues, self.eigenVectors = np.linalg.eig(self.covariance)

    def calcEigenFaces(self):
        evt = np.transpose(self.eigenVectors)[:, 0:self.numComponents]
        self.eigenFaces = np.dot(self.diffs, evt)
        self.eigenFaces /= np.linalg.norm(self.eigenFaces, axis=0)

    def calcProjections(self, persons):
        self.labels = list(map(lambda p: p.label, persons))
        self.projections = np.dot(self.eigenFaces.transpose(), self.diffs)

    def predict(self, test_data):
        diff = test_data - self.mean
        weights = np.dot(self.eigenFaces.transpose(), diff)
        distances = np.linalg.norm(self.projections.transpose()-weights, axis=1)
        min_index = np.argmin(distances)

        label = self.labels[min_index]
        confidence = distances[min_index]
        reconstruction = np.dot(self.eigenFaces, weights) + self.mean
        reconstruction_error = np.linalg.norm(test_data - reconstruction)
        return label, confidence, reconstruction_error
