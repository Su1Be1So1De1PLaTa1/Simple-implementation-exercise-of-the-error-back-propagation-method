import numpy as np

# 入力と教師データの初期化
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_ans = np.array([0, 1, 1, 0])

# バイアス項を追加した入力データ
x_bias = np.hstack((x, np.ones((x.shape[0], 1))))

# 重みの初期化
w = np.random.normal(0, 1, (3, 3))
y = np.zeros(3)
y_calc = np.zeros(4)

# 学習率
learning_rate = 0.1
# 損失の保存
loss = np.zeros(4)

# シグモイド関数とその微分
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 損失関数
def loss_func(y_cal, y_hope):
    return (y_cal - y_hope)**2

# 動作
for epoch in range(100000):
    i = np.random.randint(0, 4)
    sum1 = np.dot(x_bias[i], w[0])
    sum2 = np.dot(x_bias[i], w[1])

    y[0] = sigmoid(sum1)
    y[1] = sigmoid(sum2)
    
    sum3 = np.dot(np.append(y[:2], 1), w[2])
    y[2] = sigmoid(sum3)
    y_calc[i] = y[2]

    # 損失の計算
    loss[i] = loss_func(y[2], y_ans[i])
    
    # 進捗の出力を抑える
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {np.mean(loss)}")
    
    # 損失が十分小さいかどうかをチェック
    if np.all(loss < 1e-6):
        break

    # 逆伝播
    error = y[2] - y_ans[i]
    gradient2 = 2 * error * sigmoid_derivative(sum3)
    
    # 重みの更新
    w[2] -= learning_rate * gradient2 * np.append(y[:2], 1)

    for j in range(2):
        gradient1 = gradient2 * w[2][j] * sigmoid_derivative(sum2) if j == 1 else gradient2 * w[2][j] * sigmoid_derivative(sum1)
        w[j][:2] -= learning_rate * gradient1 * x[i]
        w[j][2] -= learning_rate * gradient1 * 1  # バイアス項の更新

# 最終的な重みを表示
print("Final Weights:")
print(w)

# 学習結果の表示
print("\nFinal Predictions:")
for i in range(4):
    sum1 = np.dot(x_bias[i], w[0])
    sum2 = np.dot(x_bias[i], w[1])

    y[0] = sigmoid(sum1)
    y[1] = sigmoid(sum2)
    
    sum3 = np.dot(np.append(y[:2], 1), w[2])
    y[2] = sigmoid(sum3)
    
    print(f"Input: {x[i]}, Prediction: {y[2]}, Expected: {y_ans[i]}")
