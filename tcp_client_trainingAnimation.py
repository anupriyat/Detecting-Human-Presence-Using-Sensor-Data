import socket
import numpy as np
import json
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation

HOST, PORT = "localhost", 9998

fig = plt.figure(figsize=(14,8))
fig.subplots_adjust(hspace=0.3,wspace=0.6)
g_temperature = fig.add_subplot(2,3,2)
g_humidity = fig.add_subplot(2,3,1)
g_light = fig.add_subplot(2,3,5)
g_co2 = fig.add_subplot(2,3,3)
g_humidityRatio = fig.add_subplot(2,3,4)

lock=threading.Lock()

list_temperature=[]
list_humidity=[]
list_light=[]
list_co2=[]
list_humidityRatio=[]
list_occupancy=[]

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
                list_humidity.append(float(data_dic['Humidity']))
                list_light.append(float(data_dic['Light']))
                list_co2.append(float(data_dic['CO2']))
                list_humidityRatio.append(float(data_dic['HumidityRatio']))
                list_occupancy.append(int(data_dic['Occupancy']))
                lock.release()
        finally:
            tcp_client.close()
            return data_dic

def drawPlot(graph, data, gname):
    graph.clear()
    x = np.arange(len(data))
    color = 'tab:green'
    graph.set_xlabel("Events", color=color)
    graph.set_ylabel(gname, color=color)
    graph.plot(x, data, color=color)
    graph.tick_params(axis='y', labelcolor=color)
    color = 'tab:blue'
    ax2 = graph.twinx()
    ax2.set_xlabel("Events", color=color)
    ax2.set_ylabel("Occupancy", color=color)
    ax2.set_ylim(bottom=-1, top=2)
    ax2.plot(x, list_occupancy, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

def animate(k):
    lock.acquire()
    drawPlot(g_temperature, list_temperature, "Temperature")
    drawPlot(g_humidity, list_humidity, "Humidity")
    drawPlot(g_light, list_light, "Light")
    drawPlot(g_co2, list_co2, "C02")
    drawPlot(g_humidityRatio, list_humidityRatio, "HumidityRatio")
    lock.release()

t = t_client_training().start()
ani=animation.FuncAnimation(fig, animate, interval=500)
plt.show()
t.join()
print("temp mean:{}".format(np.mean(list_temperature)))



