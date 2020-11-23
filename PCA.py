import numpy as np
from sklearn.preprocessing import normalize

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
        self.mean = np.mean(np.array(map(lambda p: p.data, persons)), axis=0)

    def calcDiff(self, persons):
        self.diffs = []
        for person in persons:
            self.diffs.append(person.data - self.mean)

    def calcCovariance(self):
        self.covariance = np.transpose(self.diffs) * self.diffs

    def calcEigen(self):
        self.eigenValues, self.eigenVectors = np.linalg.eig(self.covariance)

    def calcEigenFaces(self):
        evt = np.transpose(self.eigenVectors)[:, 0:self.numComponents]
        self.eigenFaces = self.diffs * evt
        self.eigenFaces = normalize(self.eigenFaces, axis=0, return_norm=False, norm='l2')

    def calcProjections(self, persons):
        self.labels = map(lambda p: p.label, persons)
        self.projections = self.diffs * np.transpose(self.eigenFaces)
