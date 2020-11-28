from PIL import Image
from numpy import asarray
from sklearn.model_selection import train_test_split
import os

from PCA import PCAEigenFace


class Person:
    id_person = None
    label = None
    data = None

    def __init__(self, id_person, label, data):
        self.id_person = id_person
        self.label = label
        self.data = data


def loadPersons(dir):
    persons = []
    labels = []
    for filename in os.listdir(dir):
        if filename.endswith('.jpg'):
            id_label = filename.split("_")
            labels.append(id_label[1])
            grayImage = Image.open(dir + "/" + filename).convert("L")
            image_array = asarray(grayImage.resize((80, 80)))
            person = Person(id_label[0], id_label[1], image_array.transpose().reshape((-1)))
            persons.append(person)
    return persons, labels


def main():
    persons, labels = loadPersons("./ORL2")
    # TODO Adicionar imagens minhas
    train_X, test_X, __, __ = train_test_split(persons, labels, stratify=labels, train_size=0.7)
    for numComponents in range(10, 21):
        model = PCAEigenFace(numComponents)
        model.train(train_X)

        correct_count = 0
        for test in test_X:
            label, confidence, reconstructionError = model.predict(test.data)
            if test.label == label:
                correct_count = correct_count + 1
        accuracy = correct_count / len(test_X) * 100
        print(str(numComponents) + ' componentes principais, acurácia: ' + str(accuracy) + '.')


if __name__ == '__main__':
    main()
