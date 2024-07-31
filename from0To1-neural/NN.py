import numpy as np
import matplotlib.pyplot as plt
import pickle
def sigmid(x):

    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):  # 上面的导数

    return sigmid(x)*(1-sigmid(x))

def mse(y_true, y_pred):

    return ((y_true - y_pred) ** 2).mean()

class OurNeauralNetwork:
    def __init__(self):
        #权重
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        #截距，bias
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x是一个有2个元素的数字数组
        h1 = sigmid(self.w1*x[0] + self.w2*x[1] + self.b1)
        h2 = sigmid(self.w3*x[0] + self.w4*x[1] + self.b2)
        o1 = sigmid(self.w5*h1 + self.w6*h2 + self.b3)

        return o1

    def train(self, data, all_y_trues):
        """
        data是一个nx2的数组，n是样本数。
        all_yrues 是一个具有n个元素的数组，元素对应data中的样本
        """
        learning_rate = 0.1
        epochs = 1000   #  # 遍历整个数据集的次数
        lossse = []
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- 前馈（feedforward）
                sum_h1 = self.w1*x[0] + self.w2*x[1] + self.b1
                h1 = sigmid(sum_h1)

                sum_h2 = self.w3*x[0] + self.w4*x[1] + self.b2
                h2 = sigmid(sum_h2)

                sum_o1 = self.w5*h1 + self.w6*h2 + self.b3
                o1 = sigmid(sum_o1)
                y_pred = o1

                #计算偏导 Naming: d_L_d_w1 represents "partial L / partial w1"
                d_l_d_ypred = -2*(y_true - y_pred)

                # Neuron 1
                d_ypred_d_w5 = h1 * derivative_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * derivative_sigmoid(sum_o1)
                d_ypred_d_b3 = derivative_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * derivative_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * derivative_sigmoid(sum_o1)


                d_h1_d_w1 = x[0] * derivative_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * derivative_sigmoid(sum_h1)
                d_h1_d_b1 = derivative_sigmoid(sum_h1)

                d_h2_d_w3 = x[0] * derivative_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * derivative_sigmoid(sum_h2)
                d_h2_d_b2 = derivative_sigmoid(sum_h2)

                #更新权重和偏差
                self.w1 -= learning_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learning_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learning_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.w3 -= learning_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learning_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learning_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                self.w5 -= learning_rate * d_l_d_ypred * d_ypred_d_w5
                self.w6 -= learning_rate * d_l_d_ypred * d_ypred_d_w6
                self.b3 -= learning_rate * d_l_d_ypred * d_ypred_d_b3


            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse(all_y_trues, y_preds)
                lossse.append(loss)
                # print("Epoch %d loss: %.3f" % (epoch, loss))

        plt.plot(range(0, epochs, 10), lossse)
        plt.title("loss over epochs")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.show()

    def save_weights(self, model_w):
        weight = {
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3,
            "w4": self.w4,
            "w5": self.w5,
            "w6": self.w6,
            "b1": self.b1,
            "b2": self.b2,
            "b3": self.b3,
        }
        with open("E:/gait/opus/model_w", "wb") as f:
            pickle.dump(weight, f)
            print("save weights to file:", model_w)

    def load_weights(self, model_w):
        with open(model_w, "rb") as f:
            weight = pickle.load(f)
            self.w1 = weight["w1"]
            self.w2 = weight["w2"]
            self.w3 = weight["w3"]
            self.w4 = weight["w4"]
            self.w5 = weight["w5"]
            self.w6 = weight["w6"]
            self.b1 = weight["b1"]
            self.b2 = weight["b2"]
            self.b3 = weight["b3"]
            print("load weights from file:", model_w)



data = np.array([
    [-2, -1],  # Alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [-15, -6],  # Dan
    [-27, -6],  # Emma
])

all_y_trues = np.array([
    1,  # Alice
    0,  # Bob
    0,  # Charlie
    1,  # Dan
    1,  # Emma
])

network = OurNeauralNetwork()
network.train(data, all_y_trues)

# network.save_weights("model_w.pkl")

guan = np.array([-7, -3]) # 128榜， 63英寸
li = np.array([20, 2])

print("Guan: %.3f" % network.feedforward(guan))  # 0.975 女F
print("Li: %.3f" % network.feedforward(li))   # 0.038 男M
