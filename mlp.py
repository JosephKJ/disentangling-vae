import torch
import os

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# train_path = '/home/josephkj_google_com/workspace/disentangling-vae/results/btcvae_mnist/training_features/'
# test_path = '/home/josephkj_google_com/workspace/disentangling-vae/results/btcvae_mnist/testing_features/'

train_path = '/home/josephkj_google_com/workspace/disentangling-vae/neo_results/btcvae_cifar10_50/training_features/'
test_path = '/home/josephkj_google_com/workspace/disentangling-vae/neo_results/btcvae_cifar10_50/testing_features/'

# train_path = '/home/josephkj_google_com/workspace/disentangling-vae/neo_results/cifar_10_vae/training_features/'
# test_path = '/home/josephkj_google_com/workspace/disentangling-vae/neo_results/cifar_10_vae/testing_features/'


# train_path = '/home/joseph/workspace/disentangling-vae/neo_results/VAE_cifar10/training_features/'
# test_path = '/home/joseph/workspace/disentangling-vae/neo_results/VAE_cifar10/testing_features/'

X = []
y = []
for file in os.listdir(train_path):
    label = file.split('_')[1].split('.')[0]
    z = torch.load(os.path.join(train_path, file)).detach().numpy()[0]
    X.append(z)
    y.append(label)

print('Loaded the data.')

clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(X, y)

print('Fit the data.')

X_test = []
y_test = []
for file in os.listdir(test_path):
    label = file.split('_')[1].split('.')[0]
    z = torch.load(os.path.join(test_path, file)).detach().numpy()[0]
    X_test.append(z)
    y_test.append(label)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(acc)
