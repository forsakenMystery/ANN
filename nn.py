import numpy as np
import matplotlib.pyplot as plot
from os import listdir
from os.path import isfile, join
import cv2 as cv

learning_rate = 1e-4

dataset="C:\\Users\\Hamed Khashehchi\\Downloads\\Telegram Desktop\\NN_Project4_1397\\faces"


def go_loading(i, total, show, r):
    percentage = i * 100 / total
    p = percentage
    s = "["
    k = 0
    while percentage > 0:
        s += "*"
        percentage -= 2
        k += 1
    k_prime = k
    while k < 50:
        k += 1
        s += " "
    if not (show == k_prime):
        # os.system('cls' if os.name == 'nt' else 'clear')
        print(
            s + "]" + str("{:2.3f}".format(p)).zfill(6) + "%  -->  " + str(i).zfill(r) + "/" + str(total).zfill(r))
    return k_prime


def go_training(i, total, show, r, train, val):
    percentage = i * 100 / total
    p = percentage
    s = "["
    k = 0
    while percentage > 0:
        s += "*"
        percentage -= 2
        k += 1
    k_prime = k
    while k < 50:
        k += 1
        s += " "
    if not (show == k_prime):
        # os.system('cls' if os.name == 'nt' else 'clear')
        print(
            s + "]" + str("{:2.3f}".format(p)).zfill(6) + "%  -->  " + str(i).zfill(r) + "/" + str(total).zfill(
                r) + " Loss Train " + str("{:1.10f}".format(train)).zfill(11) + " Loss Validation " + str(
                str("{:1.10f}".format(val)).zfill(11)))
    return k_prime


def read_dataset(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    X = []
    Y = []
    show = -1
    i = 0
    count = 0
    for f in files:
        # print(f)
        show = go_loading(i, len(files), show, 3)
        # print(cv.imread(path+"\\"+f))
        X.append(cv.imread(path+"\\"+f, 0))
        if "sunglasses" in f:
            # print(1)
            Y.append(1)
        else:
            Y.append(0)
        i += 1
    print(np.array(X).shape)
    print(np.array(Y).shape)
    # print(np.array(Y)[600:])
    # s = 0
    # l = 0
    # for i in np.array(Y)[550:600]:
    #     print(i)
    #     s+=1
    #     if i==1:
    #         l+=1
    # print(s)
    # print(l)
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
    print(X.shape)
    Y = np.array(Y)
    Y = np.reshape(Y, [Y.shape[0], 1])
    X = np.subtract(X, np.mean(X))
    X = np.divide(X, np.std(X))
    return X[0:550], Y[0:550], X[550:600], Y[550:600], X[600:], Y[600:]
# read_dataset(dataset)


class Neural_Network(object):
    def __init__(self, input_size, output_size, hidden_size):
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        # print("W1", self.W1.shape)
        # print("W2", self.W2.shape)

    def forward(self, input):
        # print("forward")
        # print("input", input.shape)
        self.z = np.dot(input, self.W1)
        # print("z", self.z.shape)
        self.z2 = self.sigmoid(self.z)
        # print("z2", self.z2.shape)
        self.z3 = np.dot(self.z2, self.W2)
        # print("z3", self.z3.shape)
        out = self.sigmoid(self.z3)
        # print("out", out.shape)
        return out

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def backward(self, input, y, out):
        # print("backward")
        # print("y", y.shape)
        # print("out", out.shape)
        # print("input", input.shape)
        self.out_error = y - out
        self.out_delta = self.out_error*self.sigmoid_derivative(out)
        self.z2_error = self.out_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoid_derivative(self.z2)
        self.W1 += learning_rate*input.T.dot(self.z2_delta)
        self.W2 += learning_rate*self.z2.T.dot(self.out_delta)

    def train(self, input, y):
        out = self.forward(input)
        self.backward(input, y, out)

    def predict(self, input, y):
        out = self.forward(input)
        count = 0
        for i in range(len(y)):
            if 0.5 < out[i] <= 1:
                tts = 1
            elif 0 <= out[i] <= 0.5:
                tts = 0
            else:
                tts = -1
            if y[i] == tts:
                count += 1
        loss = np.mean(np.square(np.subtract(y, out)))
        return 1.0*(count/len(y)), loss


def model(NN, epoch, X_train, Y_train, X_val, Y_val, X_test, Y_test):
    show = -1
    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    xx = []
    for i in range(epoch):
        xx.append(i+1)
        acc, l = NN.predict(X_train, Y_train)
        loss_train.append(l)
        acc_train.append(acc)
        # print("train")
        # print("loss", l)
        # print("accuracy", acc)
        acc, l = NN.predict(X_val, Y_val)
        loss_val.append(l)
        acc_val.append(acc)
        show = go_training(i, epoch, show, 4, loss_train[len(loss_train)-1], loss_val[len(loss_val)-1])
        # print("validation")
        # print("loss", l)
        # print("accuracy", acc)
        # print("====================================\n")
        NN.train(X_train, Y_train)

    accuracy, _ = NN.predict(X_test, Y_test)
    plot.figure()
    plot.subplot(311)
    plot.plot(xx, loss_train, 'r')
    plot.plot(xx, loss_val, 'blue')
    plot.subplot(312)
    plot.plot(xx, acc_train, 'r')
    plot.plot(xx, acc_val, 'blue')
    plot.subplot(313)
    plot.plot([epoch], [accuracy], 'bs')
    plot.show()



def main():
    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = read_dataset(dataset)
    NN = Neural_Network(960, 1, 66)
    model(NN, 10000, X_train, Y_train, X_validation, Y_validation, X_test, Y_test)


if __name__ == '__main__':
    main()
