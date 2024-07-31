from NN import *

net = OurNeauralNetwork()
net.load_weights('model_w')

sample_data = np.array([30, 1])  # 示例输入
prediction = net.feedforward(sample_data)
print("Prediction:", prediction)