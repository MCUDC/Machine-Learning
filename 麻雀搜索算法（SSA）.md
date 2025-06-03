
>麻雀算法就是一群麻雀找吃的，分了三个队伍：
>**生产者麻雀**：生产者就是离吃的最近的，找吃的到处乱飞，东一下西一下，先锋队
> **跟随者麻雀**：抱大腿，有吃的就赶紧跟上扩大优势(在好区域里，小范围多点试探，很快找出局部最优)
> **警戒者麻雀**：就是防止陷入局部最优解，拒绝做井底之雀，格局打开


##  目录


- [先上代码](#先上代码)
- [麻雀搜索算法（SSA）调优超参数详细讲解](#麻雀搜索算法ssa调优超参数详细讲解)
  - [1. 定义适应度函数（模型评价函数）](#1-定义适应度函数模型评价函数)
  - [2. 初始化参数和麻雀群](#2-初始化参数和麻雀群)
  - [3. 初始化种群（初始位置生成）](#3-初始化种群初始位置生成)
  - [4. 主迭代循环开始](#4-主迭代循环开始)
  - [5. 排序并找出最优麻雀](#5-排序并找出最优麻雀)
  - [6. 更新生产者（前20麻雀）](#6-更新生产者前20麻雀)
  - [7. 更新跟随者（普通麻雀）](#7-更新跟随者普通麻雀)
    - [跟随麻雀（Follower）位置更新详细讲解](#跟随麻雀follower位置更新详细讲解)
  - [8. 更新警戒者（特别敏感的麻雀）](#8-更新警戒者特别敏感的麻雀)
    - [解释一下加入警戒麻雀的原因](#解释一下加入警戒麻雀的原因)
  - [9. 边界控制（防止参数飞出范围）](#9-边界控制防止参数飞出范围)
  - [10. 重新评估并记录最优解](#10-重新评估并记录最优解)



# 先上代码


```py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. 适应度函数：定义评价标准
def fitness_func(params):
    n_estimators = int(params[0])  # 棵树数量
    max_depth = int(params[1])     # 最大树深度

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(train_important, train_labels)           # 用重要特征训练
    predictions = rf.predict(test_important)        # 在测试集上预测
    score = f1_score(test_labels, predictions)      # 用f1分数做适应度
    return score

# 2. 参数设定
T = 30                # 最大迭代次数
N = 20                # 麻雀个数（种群规模）
PD = int(0.2 * N)     # 生产者数量（领先探索者）
SD = int(0.1 * N)     # 侦察者数量（警觉个体）
ST = 0.8              # 安全阈值
dim = 2               # 超参数个数（n_estimators 和 max_depth）

# 参数搜索范围
lower_bound = np.array([50, 5])     # 最小值：[50棵树，5层]
upper_bound = np.array([500, 30])   # 最大值：[500棵树，30层]

# 3. 初始化麻雀种群
X = np.random.uniform(low=lower_bound, high=upper_bound, size=(N, dim))

# 4. 主循环
for t in range(T):

    # 计算每只麻雀（超参数组合）的适应度
    fitness = np.array([fitness_func(ind) for ind in X])

    # 排序（适应度高排前面）
    sort_idx = np.argsort(-fitness)
    X = X[sort_idx]
    fitness = fitness[sort_idx]

    # 记录最优最差麻雀
    x_best = X[0]
    x_worst = X[-1] #-1 是 Python 中的最后一个元素索引

    # 更新生产者（探索者）
    R2 = np.random.rand()
    for i in range(PD):
        if R2 < ST:
            X[i] = X[i] * np.exp((i + 1) / (np.random.rand() * T))  # 自我放大探索
        else:
            X[i] = X[i] + np.random.normal(0, 1, dim)              # 乱走一下看看周围

    # 更新跟随者（普通麻雀）
    for i in range(PD, N):
        A = np.random.randint(1, 3)
        if i > N/2:
            X[i] = np.random.normal(0, 1, dim) * np.exp((x_worst - X[i]) / ((i + 1)**2))
        else:
            X[i] = X[i] + A * np.abs(X[i] - x_best)

    # 更新侦察者（警觉麻雀）
    alarm_indices = np.random.choice(range(N), SD, replace=False)
    for i in alarm_indices:
        if fitness[i] > np.mean(fitness):
            X[i] = x_best + np.random.normal(0, 1, dim) * np.abs(X[i] - x_best)
        else:
            X[i] = X[i] + np.random.normal(0, 1, dim)

    # 保证每只麻雀都在参数搜索范围内
    X = np.clip(X, lower_bound, upper_bound)

# 5. 最终结果
fitness = np.array([fitness_func(ind) for ind in X])
best_index = np.argmax(fitness)
best_params = X[best_index]
print(f"最优超参数组合: n_estimators={int(best_params[0])}, max_depth={int(best_params[1])}")
print(f"最优f1分数: {fitness[best_index]:.4f}")


```


### 麻雀搜索算法（SSA）调优超参数详细讲解

---

### 1. 定义适应度函数（模型评价函数）

```python
def fitness_func(params):
    n_estimators = int(params[0])  # 棵树数量
    max_depth = int(params[1])     # 最大深度

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(train_important, train_labels)
    predictions = rf.predict(test_important)
    score = f1_score(test_labels, predictions)
    return score
```

**通俗解释：**

- 适应度函数就是打分函数。
- `params[0]` 是棵树数量，`params[1]` 是最大深度。
- 用这个超参数组合训练随机森林，最后用 F1-score 来打分。

---

### 2. 初始化参数和麻雀群

```python
T = 30                # 最大迭代次数
N = 20                # 总麻雀数量
PD = int(0.2 * N)     # 生产者数量（20%）
SD = int(0.1 * N)     # 警觉者数量（10%）
ST = 0.8              # 安全阈值

dim = 2               # 超参数个数
lower_bound = np.array([50, 5])
upper_bound = np.array([500, 30])
```

**通俗解释：**

- 迭代 30 次；
- 群体中有 20 只麻雀；
- 20% 是生产者，10% 是侦察者；
- 参数取值被限定在合理范围之内。

---

### 3. 初始化种群（初始位置生成）

```python
X = np.random.uniform(low=lower_bound, high=upper_bound, size=(N, dim))
```

**通俗解释：**

- 每只麻雀的初始位置是随机生成的；
- 形成一个 20 行 2 列的矩阵。

---

### 4. 主迭代循环开始

```python
for t in range(T):
    fitness = np.array([fitness_func(ind) for ind in X])
```

**通俗解释：**

- 每一代，让所有麻雀飞一下并计算得分；
- `fitness` 记录每只麻雀的得分情况。

---

### 5. 排序并找出最优麻雀

```python
sort_idx = np.argsort(-fitness)
X = X[sort_idx]
fitness = fitness[sort_idx]

x_best = X[0]
x_worst = X[-1]
```

**通俗解释：**

- 按得分高低重新排列麻雀位置；
- `X[0]` 是得分最高的麻雀；
- `X[-1]` 是得分最低的麻雀。

---

### 6. 更新生产者（前20%麻雀）

```python
R2 = np.random.rand()
for i in range(PD):
    if R2 < ST:
        X[i] = X[i] * np.exp((i + 1) / (np.random.rand() * T))
    else:
        X[i] = X[i] + np.random.normal(0, 1, dim)
```

**通俗解释：**

- `np.exp(x)` 的意思是：以数学常数 e 为底数，求 e 的 x 次方。
- 如果环境安全（R2 < ST），生产者继续积极探索；
- 如果不安全，生产者随机躲避。
- 简单例子：
比如原来 X[i] = 80，
随机数一搞，np.exp(...) ≈ 3.5，
那么更新后：`X[i] = 80 * 3.5 = 280`一下子跳到了很远的地方！而不是原地打转。



---

### 7. 更新跟随者（普通麻雀）

```python
for i in range(PD, N):
    A = np.random.randint(1, 3)
    if i > N/2:
        X[i] = np.random.normal(0, 1, dim) * np.exp((x_worst - X[i]) / ((i + 1)**2))
    else:
        X[i] = X[i] + A * np.abs(X[i] - x_best)
```

**通俗解释：**

- 前半部分的普通麻雀跟随最优麻雀靠近；
- 后半部分的普通麻雀远离最差麻雀，保持多样性。
#### 跟随麻雀（Follower）位置更新详细讲解

---

##### 1. 生成随机力度 A

```python
A = np.random.randint(1, 3)
```

**解释：**

- 随机生成一个整数，取值是 1 或 2；
- 控制跟随动作的力度，有时候模仿得快，有时候慢。

---

##### 2. 判断当前麻雀在种群中的位置

```python
if i > N/2:
```

**解释：**

- 判断当前麻雀是不是排在种群的后半段；
- 后半段麻雀一般表现较差，需要采取不同的策略（更多逃避、试探）。

---

##### 3. 后半段麻雀（表现较差）的更新策略

```python
X[i] = np.random.normal(0, 1, dim) * np.exp((x_worst - X[i]) / ((i + 1)**2))
```

**详细解释：**

- `np.random.normal(0, 1, dim)`：产生一个随机方向（标准正态分布，可能正也可能负）；
- `x_worst - X[i]`：计算当前位置和最差麻雀位置的差距（偏移量）；
- 除以 $(i+1)^2$：让编号越大（位置越靠后）的麻雀跳动幅度越小，更加谨慎；
- 最后整个式子乘上指数放大/缩小系数 $\exp(...)$，形成新的逃逸方向。


**举个位置靠后而且乱跑的例子方便理解**


假设你的种群里有以下几只麻雀（代表不同的超参数组合）：

| 麻雀编号 | n_estimators | max_depth | F1分数 |
|:--------:|:------------:|:---------:|:------:|
| 麻雀1    | 500          | 5         | 0.30   |
| 麻雀2    | 50           | 30        | 0.40   |
| 麻雀3    | 80           | 10        | 0.85   |


 **分析说明**

- 麻雀1 和 麻雀2 的超参数组合很差，F1分数低得离谱；
- 麻雀3 的超参数组合表现较好，F1分数接近较优水平。

**问题出现**

如果不特别处理，任由表现差的麻雀（如麻雀1、2）自由飞行：

- 它们可能随机跳到极端不合理的超参数区间；
- 例如：`n_estimators = 9000`、`max_depth = 10000`；
- 结果导致搜索过程偏离有效区域，浪费资源，降低整体收敛效率。




---




##### 4. 前半段麻雀（表现还可以）的更新策略

```python
X[i] = X[i] + A * np.abs(X[i] - x_best)
```

**详细解释：**

- `np.abs(X[i] - x_best)`：当前麻雀与最优麻雀之间的绝对距离；
- 乘以 $A$（随机取 1 或 2）：模拟不同程度的模仿；
- 加到自己的原始位置上，向最好麻雀靠近。


- 表现一般的麻雀，会以**不同的速度**直接向最优麻雀靠拢；
- 有点类似"抄作业"，快速提升自己。




---

### 8. 更新警戒者（特别敏感的麻雀）

```python
alarm_indices = np.random.choice(range(N), SD, replace=False)
for i in alarm_indices:
    if fitness[i] < np.mean(fitness): #我使用f1评分
        X[i] = x_best + np.random.normal(0, 1, dim) * np.abs(X[i] - x_best)
    else:
        X[i] = X[i] + np.random.normal(0, 1, dim)
```


- `if fitness[i] < np.mean(fitness)`:
如果这只麻雀的得分（fitness）比平均值差，说明这只麻雀位置不好，得分低，需要赶紧调整！


- 选出少数几只麻雀作为侦察者；
- 得分好的侦察者靠近最优麻雀；
- 得分差的侦察者随机逃离当前区域。

**解释一下加入警戒麻雀的原因:**
假设：

你的大部分麻雀都找到一个超参数组合，例如 `n_estimators = 150`，`max_depth = 10`，  
预测的 F1 分数是 0.80，表现已经不错了。

大家都在这个附近小幅度乱动，彼此之间的变化非常有限。

但是，警戒麻雀的作用是什么？

警戒麻雀并不会因为当前种群F1分数高就停止探索。  
它们依然按照自己的策略进行小幅度扰动，或者围绕当前最优解 `x_best` 进行随机跳动。

有时候，警戒麻雀可能跳到了非常奇怪的位置，比如：

- `n_estimators = 300`
- `max_depth = 20`

在这个新位置，它们偶然发现 F1 分数竟然提升到了 0.90！

这时，警戒麻雀会把这个新的优秀位置带回来，  
从而引导整个种群重新集中在新的更优超参数区域，  
最终让整体模型性能得到进一步提升。

这种机制保证了即使整个种群**陷入局部最优**，警戒麻雀也能凭借局部探索跳出陷阱，发现全局更优的超参数组合。

---

### 9. 边界控制（防止参数飞出范围）

```python
X = np.clip(X, lower_bound, upper_bound)
```


- np.clip(x, a, b) 的意思是：
把数组x中所有的数，强制限制在 [a, b] 这个范围内，
小于a的数，就变成a；
大于b的数，就变成b；
中间的数保持不变。
- 麻雀不能飞出超参数取值范围；
- 超出就拉回到边界值内。

---

### 10. 重新评估并记录最优解

```python
fitness = np.array([fitness_func(ind) for ind in X])
best_index = np.argmax(fitness)
best_params = X[best_index]
```



- 最后重新打分一遍；
- 找出所有麻雀中得分最高的，记录下它的超参数组合。

