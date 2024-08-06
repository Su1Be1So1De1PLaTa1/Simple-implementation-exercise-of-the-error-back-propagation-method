import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# 入力と教師データの初期化
x = np.array(rng.uniform(-1.0, 1.0, 10))
x = x.reshape(10, 1)

# バイアス項を追加した入力データ
x_bias = np.hstack((x, np.ones((x.shape[0], 1))))

# 隠れ層のノード数
hidden_nodes = 30 

# 重みの初期化
w = np.random.normal(0, 1, (2, hidden_nodes))
w_out = np.random.normal(0, 1, (hidden_nodes, 1))
print(w)
y = np.zeros(hidden_nodes + 1)
y_calc = np.zeros(10)  # 修正：10に変更

# 学習率
learning_rate = 0.1

# 損失の保存
loss = np.zeros(10)  

# hyperbolic tangent関数とその微分
def h_tangent(x):
    return (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))

def h_tangent_derivative(x):
    return (1 - h_tangent(x)) * (1 + h_tangent(x))

# 損失関数
def loss_func(y_cal, y_hope):
    return (y_cal - y_hope)**2

# 実数値関数
def a_function(input_value):
    return 0.8 * np.sin(input_value)

y_ans = np.array([a_function(input) for input in x])

# 動作
for epoch in range(100000):
    i = np.random.randint(0, 10)
    sums = np.dot(x_bias[i], w)

    for j in range(hidden_nodes):
        y[j] = h_tangent(sums[j])

    sum3 = np.dot(y[:hidden_nodes], w_out)
    y[hidden_nodes] = h_tangent(sum3)
    y_calc[i] = y[hidden_nodes]

    # 損失の計算
    loss[i] = loss_func(y[hidden_nodes], y_ans[i])
    
    # 進捗の出力
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {np.mean(loss)}")
    
    # 損失が十分小さいかどうかをチェック
    if np.all(loss < 1e-6):
        break

    # 逆伝播
    error = y[hidden_nodes] - y_ans[i]
    gradient2 = 2 * error * h_tangent_derivative(sum3)

    # 重みの更新
    de_w = learning_rate * gradient2 * y[:hidden_nodes]
    w_out -= de_w.reshape(hidden_nodes, 1)

    for j in range(hidden_nodes):
        gradient1 = gradient2 * w_out[j] * h_tangent_derivative(sums[j])
        w[:, j] -= learning_rate * gradient1 * x_bias[i]

# 最終的な重みを表示
print("重み：")
print(w)

# 学習結果の表示
print("\n最終予測結果：")
for i in range(10):  
    sums = np.dot(x_bias[i], w)
    for j in range(hidden_nodes):
        y[j] = h_tangent(sums[j])
    
    sum3 = np.dot(y[:hidden_nodes], w_out)
    y[hidden_nodes] = h_tangent(sum3)
    
    print(f"入力: {x[i]}, 予測: {y[hidden_nodes]}, 期待値: {y_ans[i]}")

# グラフの表示
# グラフ用の入力
x_input = np.arange(-1.0, 1.0, 0.001)
x_input_T = x_input.reshape(x_input.shape[0], 1)
# バイアス項つきの入力
xi_bias = np.hstack((x_input_T, np.ones((x_input_T.shape[0], 1))))

# 学習結果を用いた出力関数
def after_l_func():
    calc = []
    for i in range(xi_bias.shape[0]):  
        sums = np.dot(xi_bias[i], w)
        for j in range(hidden_nodes):
            y[j] = h_tangent(sums[j])
        
        sum3 = np.dot(y[:hidden_nodes], w_out)
        calc.append(h_tangent(sum3))
    
    return np.array(calc)

plt.plot(x_input, a_function(x_input), color = "b", label='実数値関数')
plt.plot(x_input, after_l_func(), color = "r", label='学習結果の関数')
plt.legend(prop={"family":"MS Gothic"})
plt.show()
