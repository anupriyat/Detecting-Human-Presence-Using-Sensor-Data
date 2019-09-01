import socket
import json
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import os
import pandas as pd
from time import sleep

HOST, PORT = "localhost", 9999

def test_predict(model):
    """"main function"""

    # Create a socket (SOCK_STREAM means a TCP socket)
    count = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((HOST, PORT))

        while True:
            try:

                # Receive data from the server and shut down
                received = str(sock.recv(1024), "utf-8")

                #print("Received: {}".format(received))
                if not received: break

                rec_dic=json.loads(received)

                if 'points' in rec_dic:
                    result = rec_dic
                    break
                count += 1

                X_test = pd.DataFrame.from_dict(rec_dic, orient='index').T

                X_test = X_test.drop(['Number', 'date', 'Occupancy'], axis=1)


                ypred = model.predict(X_test)

                ypred_dict = {}
                ypred_dict["Occupancy"] = int(ypred[0])
                ypred_json = json.dumps(ypred_dict)

                sock.sendall(bytes(str(ypred_json), "utf-8"))
                sleep(1)

            except:
                sock.close()
        sock.close()
    result['total'] = count
    return result


def main():

    lr = joblib.load('lr.pkl')
    dt = joblib.load('dt.pkl')
    knn = joblib.load('knn.pkl')
    lda = joblib.load('lda.pkl')
    gnb = joblib.load('gnb.pkl')
    svm = joblib.load('svm.pkl')
    rf = joblib.load('rf.pkl')
    ensemble = joblib.load('ensemble.pkl')
    crct_match_prev = 0

    models = {'Logistic Regression': lr,
              'Decision Trees': dt,
              'K-NN': knn,
              'LDA': lda,
              'GNB': gnb,
              'SVM': svm,
              'Random Forest': rf,
              'Ensemble': ensemble}

    for key, value in models.items():
        result = test_predict(value)
        crct_match = result['points']
        total = result['total']
        crct_match = crct_match-crct_match_prev
        print ("Accurate predictions for the classifier", key, ":", crct_match/total)
        crct_match_prev = crct_match


if __name__ == '__main__':
    main()

