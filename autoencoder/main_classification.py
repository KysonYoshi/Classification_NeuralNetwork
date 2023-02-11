import numpy as np
import matplotlib.pyplot as plt
import MNISTtools
import NeuralNetwork

# Digit Label to One-Hot Key
def OneHot(y):
    # --------------------------------
    #print(type(y))
    y_one_hot = np.eye(784, dtype=np.float32)[y]
    return y_one_hot
    # --------------------------------
# Compute Accuracy
def Accuracy(y,y_):
    # --------------------------------
    y_digit = np.argmax(y, 1)
    y_digit_ = np.argmax(y_, 1)
    temp = np.equal(y_digit, y_digit_).astype(np.float32)
    return np.sum(temp) / float(y_digit.shape[0])
    # --------------------------------

if __name__ == "__main__":
    # Dataset
    MNISTtools.downloadMNIST(path='MNIST_data', unzip=True)
    x_train, y_train = MNISTtools.loadMNIST(dataset="training", path="MNIST_data")
    x_test, y_test = MNISTtools.loadMNIST(dataset="testing", path="MNIST_data")

    # Show Data and Label
    print(x_train[0])
    print(y_train[0])

    # --------------------------------
    # Data Processing
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    y_train = OneHot(y_train)
    y_test_1=y_test
    y_test = OneHot(y_test)
    # --------------------------------

    # --------------------------------
    # Create NN Model
    nn = NeuralNetwork.NN(784, 128, 784, "sigmoid")
    # --------------------------------

    # Training the Model
    loss_rec = []
    batch_size = 64
    for i in range(10001):
        # --------------------------------
        # Sample Data Batch
        batch_id = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_id]
        y_batch = y_train[batch_id]
        # --------------------------------

        # --------------------------------
        # Forward & Backward & Update
        nn.feed({"x": x_batch, "y": y_batch})
        nn.forward()
        nn.backward()
        nn.update(1e-2)
        # --------------------------------

        # --------------------------------
        # Loss
        loss = nn.computeLoss()
        loss_rec.append(loss)
        # --------------------------------

        # --------------------------------
        # Evaluation
        batch_id = np.random.choice(x_test.shape[0], batch_size)
        x_test_batch = x_test[batch_id]
        y_test_batch = y_test[batch_id]
        nn.feed({"x": x_test_batch})
        y_test_out = nn.forward()
        acc = Accuracy(y_test_out, y_test_batch)
        if i % 100 == 0:
            print("\r[Iteration {:5d}] Loss={:.4f}".format(i, loss))
        # --------------------------------

    nn.feed({"x":x_test})
    y_prob = nn.forward()

    fig = plt.figure(figsize=(16, 8))
    columns = 16
    rows = 3
    for i in range(16):
        img = x_test[i].reshape((28,28))
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img, cmap='gray')
    for i in range(16):
        img = nn.W1[:,i].reshape((28, 28))
        fig.add_subplot(rows, columns, i+17)
        plt.imshow(img, cmap='gray')
    for i in range(16):
        img = y_prob[i].reshape((28, 28))
        fig.add_subplot(rows, columns, i+33)
        plt.imshow(img, cmap='gray')
    plt.show()
