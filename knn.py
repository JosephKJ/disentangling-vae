import torch
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_path = '/home/josephkj_google_com/workspace/disentangling-vae/neo_results/cifar_10_vae/training_features/'
test_path = '/home/josephkj_google_com/workspace/disentangling-vae/neo_results/cifar_10_vae/testing_features/'

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

clf = KNeighborsClassifier(n_neighbors=3)
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
