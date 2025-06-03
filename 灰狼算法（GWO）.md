>灰狼算法就是你是一个小狼没找到吃的快饿死了，然后有三个大佬狼找到了，你就赶紧去找他们，想办法一点一点缩小范围找到他们，就是找出最优超参数


# 先上代码
```py
import numpy as np
from sklearn.ensemble import RandomForestClassifier #我的任务是分类任务
from sklearn.metrics import f1_score
import random

# ------------------- 基础设置 -------------------

# 你要优化的超参数范围
param_bounds = {
    'n_estimators': (100, 1000),
    'max_depth': (5, 30),
    'min_samples_split': (2, 10)
}

# 灰狼参数
n_wolves = 5   # 狼的数量
n_iter = 3     # 迭代次数（这里示例用3次，实际可以30次以上）
a_decay = 2 / n_iter

# ------------------- 初始化 -------------------

wolves = []
for _ in range(n_wolves):
    wolf = [
        random.uniform(*param_bounds['n_estimators']),
        random.uniform(*param_bounds['max_depth']),
        random.uniform(*param_bounds['min_samples_split'])
    ]
    wolves.append(wolf)

wolves = np.array(wolves)

# 适应度评价函数
def fitness(wolf):
    n_estimators = int(wolf[0])
    max_depth = int(wolf[1])
    min_samples_split = int(wolf[2])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_important, train_labels)
    preds = model.predict(test_important)
    return f1_score(test_labels, preds)

# ------------------- 灰狼主循环 -------------------

for iter_num in range(n_iter):
    scores = np.array([fitness(wolf) for wolf in wolves])
    
    # 排序，找到α、β、δ
    indices = np.argsort(-scores)
    alpha, beta, delta = wolves[indices[0]], wolves[indices[1]], wolves[indices[2]]
    
    a = 2 - iter_num * a_decay
    
    for i in range(n_wolves):
        for j in range(wolves.shape[1]):
            r1, r2 = random.random(), random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha[j] - wolves[i, j])
            X1 = alpha[j] - A1 * D_alpha

            r1, r2 = random.random(), random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta[j] - wolves[i, j])
            X2 = beta[j] - A2 * D_beta

            r1, r2 = random.random(), random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta[j] - wolves[i, j])
            X3 = delta[j] - A3 * D_delta

            wolves[i, j] = (X1 + X2 + X3) / 3

    # 把超参数裁剪回合理范围
    for wolf in wolves:
        wolf[0] = np.clip(wolf[0], *param_bounds['n_estimators'])
        wolf[1] = np.clip(wolf[1], *param_bounds['max_depth'])
        wolf[2] = np.clip(wolf[2], *param_bounds['min_samples_split'])

# 最优狼位置
best_wolf = wolves[np.argmax([fitness(wolf) for wolf in wolves])]
print("\n最优超参数：")
print(f"n_estimators: {int(best_wolf[0])}, max_depth: {int(best_wolf[1])}, min_samples_split: {int(best_wolf[2])}")

```


---

# GWO（灰狼算法）调优超参数详细讲解



## 1. 整体故事版理解

想象一个故事：

你有一片大草原，每只狼，都带着自己的一套超参数（比如：种树多少棵？树多高？节点怎么分？）

你让每只狼去测试超参数，看谁能造出最准的分类器（F1 分数高）。

然后，让得分最高的 3 只狼（α、β、δ）当“头狼”。

其他狼围着这 3 只头狼，慢慢地、聪明地调整自己的超参数（不是盲目冲刺，而是带有随机性的靠近）。

经过一轮又一轮，狼群越来越接近最优超参数，最终选出最好的那个！

---

## 2. 代码结构总览

| 步骤             | 具体做什么               | 目的                           |
|------------------|---------------------------|--------------------------------|
| 定义超参数范围    | 设定超参数的合法取值区间    | 防止超参数乱飞                  |
| 初始化狼群        | 随机生成一群超参数组合      | 形成初始探索                    |
| 适应度函数（fitness）| 训练+预测+F1打分        | 测试每组超参数好不好              |
| 灰狼主循环        | 选出前三名、带动其他狼调整  | 群体智能搜索                     |
| 边界修正          | 超出合法区间就拉回来         | 保持参数合理                     |
| 选出最优解        | 找出分数最高的一只狼        | 得到最终超参数                   |

---

## 3. 每一部分更详细解释

### 3.1 基础设置（超参数范围和灰狼参数）

```python
param_bounds = {
    'n_estimators': (100, 1000),
    'max_depth': (5, 30),
    'min_samples_split': (2, 10)
}
```

含义解释：

| 超参数             | 意义                 | 范围        |
|-------------------|----------------------|------------|
| `n_estimators`    | 森林里有多少棵树       | 100 ~ 1000 |
| `max_depth`       | 每棵树最大深度         | 5 ~ 30     |
| `min_samples_split` | 一个节点最少多少样本才能分叉 | 2 ~ 10     |

灰狼参数设定：

- 5只狼（5组超参数组合）
- 3次迭代（逐步靠近最优）
- `a_decay` 控制跳跃范围逐渐减小

---

### 3.2 初始化狼群

```python
wolves = []
for _ in range(n_wolves):
    wolf = [
        random.uniform(*param_bounds['n_estimators']),
        random.uniform(*param_bounds['max_depth']),
        random.uniform(*param_bounds['min_samples_split'])
    ]
    wolves.append(wolf)
```

假设初始化后：

| 狼编号 | n_estimators | max_depth | min_samples_split |
|--------|--------------|-----------|-------------------|
| 1      | 450          | 20        | 5                 |
| 2      | 700          | 10        | 3                 |
| 3      | 800          | 15        | 2                 |
| 4      | 600          | 25        | 7                 |
| 5      | 300          | 12        | 4                 |

每只狼随机起步，各自出发寻找最佳超参数组合。

---

### 3.3 适应度评价函数 `fitness()`

```python
def fitness(wolf):
    n_estimators = int(wolf[0])
    max_depth = int(wolf[1])
    min_samples_split = int(wolf[2])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_important, train_labels)
    preds = model.predict(test_important)
    return f1_score(test_labels, preds)
```

**说明：**

- 用每只狼的超参数训练模型；
- 在测试集上预测；
- 用 F1 分数打分；
- 分数高说明超参数好。

---

### 3.4 主循环（每一轮灰狼狩猎）
```py
for i in range(n_wolves):
        for j in range(wolves.shape[1]):
```
- 这是取第 1 维的长度（下标从0开始）：

- `wolves.shape[0]` → 行数 → 狼的数量（n_wolves）

- `wolves.shape[1]` → 列数 → 每只狼的参数个数

**一轮流程：**
```py
  scores = np.array([fitness(wolf) for wolf in wolves])
    
    # 排序，找到α、β、δ
    indices = np.argsort(-scores)
    alpha, beta, delta = wolves[indices[0]], wolves[indices[1]], wolves[indices[2]]
```
1.每只狼计算 fitness 分数；
2. 找出得分前 3 名：
   - 第一名 α
   - 第二名 β
   - 第三名 δ
3. 更新每只狼的位置：
```py
 		r1, r2 = random.random(), random.random()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha[j] - wolves[i, j])
        X1 = alpha[j] - A1 * D_alpha
```

**更新公式这段展开讲一下（通俗废话版）**

注：公式参照灰狼算法数学模型

- ① `r1, r2 = random.random(), random.random()`
  - 随机生成两个 0 到 1 之间的小数，引入随机性；保证狼群每次更新有差异，不死板；有助于探索不同方向，防止陷入局部最优。
-  ② `A1 = 2 * a * r1 - a`

   -  `a` 是一个随时间迭代线性下降的参数（从 2 到 0）； `r1` 是随机数；-所以 `A1` 的范围是 `[-a, +a]`。

   - `A1` 控制了位置更新的“弹性”。
    - `A1` 小，狼慢慢靠近猎物；
    - `A1` 大，狼可能跳跃一大步。
   - 如果 `A1` 为负值，狼可能朝猎物反方向跳，有助于避免陷入局部最优。



- ③ `C1 = 2 * r2`

  - `C1` 是简单地取 `2` 倍随机数，引入偏差；




- ④ `D_alpha = abs(C1 * alpha[j] - wolves[i, j])`

  - 计算当前第 `i` 只狼到猎物（α狼）在第 `j` 个特征方向上的“感知距离”。


  - 引入 `C1`，模拟了不同狼感知偏差后的距离感知。



- ⑤ `X1 = alpha[j] - A1 * D_alpha`

  - 这是实际的位置更新公式。狼不会直接跳到猎物身上而是参考猎物方向，再加上一点随机的弹跳调整；随着迭代，狼逐步接近猎物的位置，但又有探索性，避免陷入局部最优。

- 计算往 α、β、δ 三个方向靠近的中间位置；
- 最终新位置是三者平均。

**为什么要三方平均？**

- 防止只参考一只头狼，容易出问题；


**更新位置细节公式（每个超参数一维一维更新）：**

- 计算感知距离（D_alpha，D_beta，D_delta）；
- 加入随机扰动（A1, C1）；
- 更新新位置。

这种机制模拟了现实动物行为——带点犹豫、试探，不盲目直冲。

#### 灰狼算法一维特征更新详细过程讲解（以 max_depth 为例）



 **场景设定**

只看一个超参数 `max_depth`（树的最大深度）。

假设：

- 当前狼的位置值：15
- α狼的位置值：20
- β狼的位置值：25
- δ狼的位置值：10

---

**开始更新步骤**

每只狼更新每一维度，要经历三次靠拢（分别向 α、β、δ）并取平均。



**1. 向 α 靠拢**

当前值：15  
α狼值：20

随机生成两个随机数：

```python
r1 = 0.6
r2 = 0.7
```

计算：

```python
A1 = 2 * a * r1 - a
C1 = 2 * r2
```

假设当前 `a = 1.5`：

```python
A1 = 2 * 1.5 * 0.6 - 1.5 = 0.3
C1 = 2 * 0.7 = 1.4
```

计算感知到的距离：

```python
D_alpha = abs(C1 * alpha - 当前值)
D_alpha = abs(1.4 * 20 - 15) = abs(28 - 15) = 13
```

更新位置：

```python
X1 = alpha - A1 * D_alpha
X1 = 20 - 0.3 * 13 = 20 - 3.9 = 16.1
```

所以，向 α 狼靠拢得到的新位置为 **16.1**。

---

**2. 向 β 靠拢**

当前值：15  
β狼值：25

新的随机数：

```python
r1 = 0.3
r2 = 0.4
```

计算：

```python
A2 = 2 * 1.5 * 0.3 - 1.5 = -0.6
C2 = 2 * 0.4 = 0.8
```

感知到的距离：

```python
D_beta = abs(0.8 * 25 - 15) = abs(20 - 15) = 5
```

更新位置：

```python
X2 = beta - A2 * D_beta
X2 = 25 - (-0.6) * 5
X2 = 25 + 3 = 28
```

所以，向 β 狼靠拢得到的新位置为 **28**。

---

**3. 向 δ 靠拢**

当前值：15  
δ狼值：10

新的随机数：

```python
r1 = 0.9
r2 = 0.2
```

计算：

```python
A3 = 2 * 1.5 * 0.9 - 1.5 = 1.2
C3 = 2 * 0.2 = 0.4
```

感知到的距离：

```python
D_delta = abs(0.4 * 10 - 15) = abs(4 - 15) = 11
```

更新位置：

```python
X3 = delta - A3 * D_delta
X3 = 10 - 1.2 * 11
X3 = 10 - 13.2 = -3.2
```

所以，向 δ 狼靠拢得到的新位置为 **-3.2**。

---

**4. 三个结果取平均**

最终新位置为：

```python
新的位置 = (X1 + X2 + X3) / 3
新的位置 = (16.1 + 28 + (-3.2)) / 3
新的位置 = 40.9 / 3 ≈ 13.63
```

所以，最终更新后新的 `max_depth` 大约是 **13.63**，是浮点数，所以在打分之前，需要 int() 转换成整数` n_estimators = int(wolf[0])`

（当然之后一般会使用 `clip` 函数，限制在超参数合法范围内，例如 5~30。）

---

**更新过程总结表格**

| 向谁靠拢 | 随机数 (r1, r2) | 中间计算          | 新位置 |
|----------|-----------------|-------------------|--------|
| α狼 (20) | r1=0.6, r2=0.7   | X1=16.1            | 16.1   |
| β狼 (25) | r1=0.3, r2=0.4   | X2=28              | 28     |
| δ狼 (10) | r1=0.9, r2=0.2   | X3=-3.2            | -3.2   |
| 平均后    | -               | (16.1+28-3.2)/3=13.63 | 13.63 |

---

**直观理解**

想象当前狼的位置在 15，随后：

- 靠 α狼 → 稍微向右靠 → 16.1
- 靠 β狼 → 用力向右靠 → 28
- 靠 δ狼 → 反向大幅向左跳 → -3.2

最后综合三次动作取平均，最终来到 13.63，稍微比原本 15 略微向左偏移。


---

### 3.5 超参数范围修正

```python
wolf[0] = np.clip(wolf[0], *param_bounds['n_estimators'])
```

**说明：**

- 如果位置超出了设定范围，比如 `n_estimators > 1000`；
- 则用 `clip` 方法拉回合法区间。

避免出现不合理参数导致训练失败或异常。



---




### 3.6 选出最优狼

全部迭代结束后：

```python
best_wolf = wolves[np.argmax([fitness(wolf) for wolf in wolves])]
```

再打一遍分，找出得分最高的那只狼。

最终输出最佳超参数组合：

```
最优超参数：
n_estimators: 789, max_depth: 23, min_samples_split: 4
```

---


##  4.灰狼优化算法（GWO）数学建模全流程



### 1. 灰狼社会结构建模

在 GWO 中，灰狼被分成四个角色：

| 等级   | 名称   | 代表意义      |
|-------|--------|--------------|
| 第一   | α狼    | 最好的解，领导者 |
| 第二   | β狼    | 次好的解，辅助α狼 |
| 第三   | δ狼    | 第三好的解，协助α和β |
| 其他   | ω狼    | 普通成员，跟随头狼行动 |

**数学意义：**

- 每只狼的位置对应一个解（如超参数组合）。
- 狼的位置是一个向量：

$$
X = [x_1, x_2, \dots, x_n]
$$

其中 \( n \) 是问题的维度，比如超参数数量就是 3（如 n_estimators、max_depth、min_samples_split）。

---

### 2. 猎物包围行为建模（Encircling Prey）

狼群包围猎物并尝试靠近猎物位置，但猎物的真实位置未知，通常用 α、β、δ 三头狼的位置来估计。

**数学建模公式：**

- **距离向量（Distance vector）：**


$$
D = |C \cdot X_p - X|
$$

其中：

- $X_p$：猎物位置（由 α、β、δ估计）
- $X$：当前狼的位置
- $C$：扰动系数向量

- **位置更新公式（Position update）：**

$$
X(t+1) = X_p - A \cdot D
$$

其中：

- $A$：搜索方向和幅度控制向量
- $t$：当前迭代步

**小解释：**

- $A$ 和 $C$ 都是动态变化的；
- 引入随机性，增加探索性；
- $A$ 可以为负数，允许狼群反方向移动，增强全局搜索能力。

---

### 3. 系数向量 A 和 C 的生成


**公式：**

$$
A = 2ar_1 - a
$$

$$
C = 2r_2
$$

其中：

- $a$：从 2 线性下降到 0（控制搜索范围逐渐减小）
- $r_1, r_2$：均匀分布的随机数（取值范围 0 到 1）

**总结：**

- $A$：控制搜索强度，前期跳得远，后期跳得小；
- $C$：控制扰动幅度，增加探索能力。

---

### 4. 猎物位置估计和综合更新（Hunting）


由于猎物位置不确定，GWO 使用 α、β、δ 三头狼综合估计猎物大致位置。

**具体步骤：**

- 向 α 狼靠拢：

$$
D_\alpha = |C_1 \cdot X_\alpha - X|
$$

$$
X_1 = X_\alpha - A_1 \cdot D_\alpha
$$

- 向 β 狼靠拢：

$$
D_\beta = |C_2 \cdot X_\beta - X|
$$

$$
X_2 = X_\beta - A_2 \cdot D_\beta
$$

- 向 δ 狼靠拢：

$$
D_\delta = |C_3 \cdot X_\delta - X|
$$

$$
X_3 = X_\delta - A_3 \cdot D_\delta
$$

- **综合更新最终位置：**

$$
X(t+1) = \frac{X_1 + X_2 + X_3}{3}
$$

即，取向 α、β、δ 狼靠拢后的平均作为新的位置。

---

### 小结

| 阶段         | 数学公式                               | 主要意义              |
|--------------|----------------------------------------|----------------------|
| 包围猎物     | $D = C \cdot X_p - X$                   | 估计与猎物的距离       |
| 移动更新     | $X(t+1) = X_p - A \cdot D$               | 向猎物靠近             |
| A系数生成    | $A = 2ar_1 - a$                          | 控制搜索强度和方向      |
| C系数生成    | $C = 2r_2$                               | 加入扰动，模拟感知偏差   |
| 位置融合更新 | $X(t+1) = \frac{X_1 + X_2 + X_3}{3}$    | 综合三狼的推测，更新位置 |


---



## 5. 更新直观示意图

每轮迭代狼群位置示意：

```
Round 1：
🐺🐺🐺🐺🐺     → α
  ↘ ↘ ↘
Round 2：
🐺🐺🐺    → α
 ↘ ↘
Round 3：
🐺🐺 → α
 ↘
最终只剩冠军狼
```

- 随着轮次增加，狼群逐渐向最优方向聚集。



