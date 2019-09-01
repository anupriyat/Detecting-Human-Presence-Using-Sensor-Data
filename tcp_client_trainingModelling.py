import socket
import numpy as np
import json
import threading
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

import os

HOST, PORT = "localhost", 9998

lock=threading.Lock()

list_temperature=[]
list_humidity=[]
list_light=[]
list_co2=[]
list_humidityRatio=[]
list_occupancy=[]
list_date = []

class t_client_training(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        self.work_with_server()

    def work_with_server(self):
        tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            tcp_client.connect((HOST, PORT))
            while True:
                received = str(tcp_client.recv(1024), "utf-8")
                if not received:
                    break
                print("Received: {}".format(received))
                data_dic=json.loads(received)
                lock.acquire()
                list_temperature.append(float(data_dic['Temperature']))
                #list_date.append((data_dic['date']))
                list_humidity.append(float(data_dic['Humidity']))
                list_light.append(float(data_dic['Light']))
                list_co2.append(float(data_dic['CO2']))
                list_humidityRatio.append(float(data_dic['HumidityRatio']))
                list_occupancy.append(int(data_dic['Occupancy']))
                lock.release()
        finally:
            tcp_client.close()

def get_models():
    """"Generate a library of base learners"""
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100, random_state=123)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=123)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=123)
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=123)
    models = [('svm', svc),
                ('knn', knn),
                ('naive bayes', nb),
                ('mlp-nn', nn),
                ('random forest', rf),
                ('gbm', gb),
                ('logistic', lr)]

    return models


def main():
    # t = t_client_training()
    # t.start()
    # t.join()
    #
    # df = pd.DataFrame(
    #     list(zip(list_temperature, list_humidity, list_light, list_co2, list_humidityRatio, list_occupancy)),
    #     columns=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy'])
    #
    # df.to_csv(ctrain_check, encoding='utf-8', index=False)

    cwd = os.getcwd()
    df = pd.read_csv(cwd+'/occupancy/censortraining.csv', index_col=False)
    df = df.drop(['number','date'], axis=1)

    X_train = df.loc[:, df.columns != 'Occupancy']
    y_train = df['Occupancy']

    lr = LogisticRegression().fit(X_train, y_train)
    joblib.dump(lr, 'lr.pkl')
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
          .format(lr.score(X_train, y_train)))

    dt = DecisionTreeClassifier().fit(X_train, y_train)
    joblib.dump(dt, 'dt.pkl')
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
          .format(dt.score(X_train, y_train)))

    knn = KNeighborsClassifier().fit(X_train, y_train)
    joblib.dump(knn, 'knn.pkl')
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(X_train, y_train)))

    lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
    joblib.dump(lda, 'lda.pkl')
    print('Accuracy of LDA classifier on training set: {:.2f}'
          .format(lda.score(X_train, y_train)))

    gnb = GaussianNB().fit(X_train, y_train)
    joblib.dump(gnb, 'gnb.pkl')
    print('Accuracy of GNB classifier on training set: {:.2f}'
          .format(gnb.score(X_train, y_train)))

    svm = SVC(gamma='scale').fit(X_train, y_train)
    joblib.dump(svm, 'svm.pkl')
    print('Accuracy of SVM classifier on training set: {:.2f}'
          .format(svm.score(X_train, y_train)))

    rf = RandomForestClassifier(max_features=3, n_estimators=100, random_state=123).fit(X_train, y_train)
    joblib.dump(rf, 'rf.pkl')
    print('Accuracy of Random Forest classifier on training set: {:.2f}'
          .format(svm.score(X_train, y_train)))

    estimators = get_models()
    ensemble = VotingClassifier(estimators, voting='hard').fit(X_train, y_train)
    joblib.dump(ensemble, 'ensemble.pkl')
    print('Accuracy of ensemble classifier on training set: {:.2f}'
          .format(ensemble.score(X_train, y_train)))

if __name__ == '__main__':
    main()
