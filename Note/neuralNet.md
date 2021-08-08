#神经网络

**机器学习有两宝，一是算法，二是特征选择。**

实际上，选取特征是更关键的一环，选取特征需要人类对问题有深刻理解，经验和思考。但是机器学习下的两个分支(神经网络算法-深度学习，reward/penalty机制-强化学习)希望脱离人的控制，自动学习应该提取什么特征，完成选取特征这个问题。

**神经网络：**
结构：多个*单独的单元*按照*一定规则*相互**连接**。
训练算法：**反向传播算法**。

##单独的单元

机器学习的感知器或线性单元，神经网络的神经元。
*区别：*
**激活函数：** 神经元选择 sigmoid\tanh函数。
$$sigmoid(x)=\frac{1}{1+e^{-x}}
$$$$tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}
$$

**sigmoid函数**是一个非线性函数，值域是（0，1）。
sigmoid函数的导数是  *y'=y(1-y)*。
**tanh函数**也是一个线性函数，值域是（-1，1），奇函数。
tanh函数的导数是  *y'=1-y^2*

##数学推导：
$$(\frac{u}{v})'=\frac{u'v-uv'}{v^2}
$$$$
(\frac{1}{1+e^{-x}})'=\frac{e^{-x}}{(1+e^{-x})^2}=(1-\frac{1}{1+e^{-x}})\frac{1}{1+e^{-x}}
$$$$
\frac{(e^{x}-e^{-x})'(e^{x}+e^{-x})-(e^{x}-e^{-x})(e^{x}+e^{-x})'}{(e^{x}+e^{-x})^2}=\frac{(e^{x}+e^{-x})^2-(e^{x}-e^{-x})^2}{(e^{x}+e^{-x})^2}=1-(\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}})^2
$$
##一定的规则

模型的训练规则往往是激活函数的不同、目标值接近程度的不同、优化算法的不同、模型层数的不同、网络模型结构的不同。
但求训练规则的方式相同，采用链式推导法求解。

###FC(Full Connection) 全连接神经网络

神经网络是一种模型，和感知器/线性单元等独立单元相区别的是：
神经网络模型是多个神经元组成的，如果对神经元比较模糊，就先当成感知器吧！

看看全连接神经网络的连接规则。
神经网络模型比简单的感知器、线性单元要复杂一点，由3个部分组成：输入层，隐藏层，输出层。

**FC作为一种神经网络模型，规则是：
1、神经元按照层分类，最左边的层叫做输入层，负责接收输入数据；最右边的层叫输出层，我们可以从这层获取神经网络输出数据。输入层和输出层之间的层叫做隐藏层，因为它们对于外部来说是不可见的，层的神经元个数是超参数；
2、每一层的内部神经元没有任何连接；
3、每一层的全部神经元和下一层（除输出层）的每个神经元都连接，前一层神经元的输出就是后一层神经元的输入（全连接的由来）；
4、每个全连接都有权重项和偏置项，用vector(w)表示**
输入层和感知器的输入没有差别。
隐藏层是用来计算的，可以视神经网络是多层感知器的计算处理，一个隐藏层意味着一组感知器计算，神经网络到最后的输出层结束一轮迭代。全连接神经网络比简单的感知器、线性单元计算效果应该更优秀，至少从规则设计上可以得到这样的结论。(更多的计算处理，更快的收敛。)


#神经网络
##模型

神经网络模型实际上就是一个输入向量到输出向量的函数表达。
$$
\vec{y}=f_{neuralNet}\vec{x}=f(W*\vec{x})
$$$$
\vec{a}=sigmoid(W1*\vec{x}) 
$$$$\vec{y}=sigmoid(W2*\vec{a})
$$

当我们认为神经网络是一个模型时，权值就是模型的**参数**，也就是模型要学习的东西。然而，一个神经网络的连接方式、网络的层数、每层的节点数这些参数，则**不是学习出来的**，而是人为事先设置的。对于**这些人为设置的参数**，我们称之为**超参数**(Hyper-Parameters)。


##目标函数

目标函数指的是输出值和目标值的接近程度

一个样本的误差：对应线性单元的ei，
是神经网络输出层的**节点和**。

目标函数如果仍采纳线性单元的规则，可以整理成：

$$
    e=\frac{1}{2}\sum_{i\in outputs}(t_i-y_i)^2
$$

##优化算法

*一阶*随机梯度下降算法
$$
W_{new_{ji}}=W_{old_{ji}}-\eta\frac{\partial e}{\partial W_{ji}}
$$
对于目标函数而言，变量仅仅是Wji, ti和xi的值都是有监督学习给出的样本数值，对于模型的训练主要是权重项和偏置项，在优化算法中也就是Wji的训练。

**推导：**
$$
e=\frac{1}{2}\sum_{i\in{outputs}}(t_i-y_i)^2
$$$$
y_i=f(\vec W_{ia}*\vec a)=f(\vec W_{ia}*f(\vec W_{ax}*\vec x))
$$

随机梯度下降算法也需要求出误差对于每个权重的偏导数（也就是梯度）。更新每一个权重项。

Wji怎么影响e呢？

观察神经网络模型，能发现Wji是通过影响节点j来影响网络中的结构，即 Wji*xi。

机器学习中将影响节点j的部分称为netj,中文叫节点j的加权输入。$$
net_j=\sum_{i\in{connect_j}}Wji*xi
$$ 

$$
\frac{\partial e}{\partial W_{ji}}=\frac{\partial e}{\partial net_j}*\frac{\partial net_j}{\partial W_{ji}}
$$

通过模型算法发现：
神经网络的激活函数和线性单元的激活函数不同，线性单元的激活函数导数是1，sigmoid函数的导数是y(1-y),tanh函数的导数是1-y^2。

重申一下，梯度下降是神经网络模型的计算值和真实值的接近程度，即目标函数。
神经网络模型的计算值是和神经网络的输出层紧密联系的，在计算有关权重项的更新时，都需要依赖输出层的节点值。所以我们又将计算各个节点连接的权重项（神经网络的训练算法）称之为**反向传播算法**。

神经网络的训练算法对于输出层节点和隐藏层节点的网络误差计算公式不同，可能就这是为什么将节点用输入层、隐藏层、输出层命名的原因吧！！！

首先看输出层节点：
输出层和目标函数直接挂钩。$$
\frac{\partial{e}}{\partial{W_{ji}}}=\frac{\partial{e}}{\partial{y_j}}*\frac{\partial{y_j}}{\partial{net_j}}*\frac{\partial{net_j}}{\partial{W_{ji}}}
$$$$
\frac{\partial{e}}{\partial{y_j}}=\frac{\partial{1/2\sum_{j\in{outputs}}}(t_j-y_j)^2}{\partial{y_j}}=-(t_j-y_j)
$$$$
\frac{\partial{y_j}}{\partial{net_j}}=\frac{\partial{sigmoid(net_j)}}{\partial{net_j}}=y_j(1-y_j)
$$
###or
$$
\frac{\partial{y_j}}{\partial{net_j}}=\frac{\partial{tanh(net_j)}}{\partial{net_j}}=1-{y_i}^2
$$$$
\frac{\partial{net_j}}{\partial{W_{ji}}}=\frac{\partial{\sum_{i\in{Upstream(i)}}W_{ji}*X_{ji}}}{\partial{W_{ji}}}=x_{ji}
$$
综上，最后得到输出层节点的网络误差：
$$\delta{j}=\frac{\partial{e}}{\partial{net_j}}=-(t_j-y_j)y_j(1-y_j)
$$
因为前存在符号，所以神经网络模型将训练算法的网络误差$\delta_j=-\frac{\partial{e}}{\partial{net_j}}$,结果也变成$\delta_j=(t_j-y_j)y_j(1-y_j)$ or $\delta_j=(t_j-y_j)(1-{y_j}^2)$。

最后是隐藏层节点:
隐藏层节点的输入权重不会直接影响输出单元，但是会间接通过节点值影响后面的输入权重，从而达到影响输出单元的目的。
隐藏层节点的网络误差是由隐藏层节点的下游网络误差反向推导而来的。如：我们可以很轻松的得到输出层的网络误差计算公式,所以得到隐藏节点的网络误差也是可以根据已经计算得到的下游节点网络误差推导出来的。
$$\frac{\partial{e}}{\partial{net_j}}=\frac{\partial{e}}{\partial{net_k}}*\frac{\partial{net_k}}{\partial{net_j}}=\frac{\partial{e}}{\partial{net_k}}*\frac{\partial{net_k}}{\partial{a_j}}*\frac{\partial{a_j}}{\partial{net_j}}
$$
$net_k$是与节点j有关的所有下游节点的相关输入权重，因为全连接神经网络节点j会影响所有下游节点，所有计算$\frac{\partial{e}}{\partial{a_j}}$时使用全导数公式,$k\in{DownStream(j)}$。
$$
\frac{\partial{e}}{\partial{net_k}}=-\delta{k}
$$$$
\frac{\partial{net_k}}{\partial{a_j}}=w_{kj}
$$$$
\frac{\partial{a_j}}{\partial{net_j}}=\frac{{\partial{sigmoid(net_j)}}}{\partial{net_j}}=a_j(1-a_j)
$$

综上，可以得到隐藏层节点的网络误差$\delta{j}=\frac{\partial{e}}{\partial{net_j}}=a_j(1-a_j)*\sum_{k\in{DownStream(j)}}(w_{kj}*-\delta{k})$。

具体推导：
$$
\begin{align}
\frac{\partial{E_d}}{\partial{net_j}}&=\sum_{k\in Downstream(j)}\frac{\partial{E_d}}{\partial{net_k}}\frac{\partial{net_k}}{\partial{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_k\frac{\partial{net_k}}{\partial{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_k\frac{\partial{net_k}}{\partial{a_j}}\frac{\partial{a_j}}{\partial{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_kw_{kj}\frac{\partial{a_j}}{\partial{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_kw_{kj}a_j(1-a_j)\\
&=-a_j(1-a_j)\sum_{k\in Downstream(j)}\delta_kw_{kj}
\end{align}
$$

统一$\delta{j}=-\frac{\partial e}{\partial net_j}=a_j(1-a_j)\sum_{k\in Downstream(j)}\delta_kw_{kj}$。
隐藏层的推导满足偏导数的复合函数链式求导法，根据隐藏层节点得到的网络误差可以很清楚的得知节点的网络误差依赖下游节点的网络误差，无论下游节点是隐藏层节点还是输出层节点，都需要先得到后面的，才能依次得到前面的，所以计算权重项很明显从后往前依次计算，这也是反向传播算法的定义和具体表现。

有了网络误差，可以轻松得出权重项的迭代更新：
$$
\begin{align}
w_{ji}\leftarrow w_{ji}-\eta*\frac{\partial{e}}{\partial{W_{ji}}}\\
\end{align}
$$
对于输出节点：
$$
w_{ji}=w_{ji}+\eta*\delta j*x_{ji}=w_{ji}+\eta*(t_j-y_j)y_j(1-y_j)*x_{ji}
$$
对于隐藏层节点：
$$
w_{ji}=w_{ji}+\eta*a_j(1-a_j)\sum_{k\in{DownStream(j)}}(w_{kj}*\delta_k)*x_{ji}
$$

本篇神经网络模型的训练规则是根据激活函数是sigmoid函数、平方和误差、全连接网络、随机梯度下降优化算法。如果激活函数不同、误差计算方式不同、网络连接结构不同、优化算法不同，则具体的训练规则也会不一样。但是无论怎样，训练规则的推导方式都是一样的，应用链式求导法则进行推导即可。







