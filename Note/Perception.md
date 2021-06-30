<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
#感知器的理论之家

##感知器是什么
###组成部分：
* 输入权值 感知器可以接收多个输入，并在输入上挂对应的权值，并且附添一个偏置项b,并设置对应的默认输入是1
* 激活函数 计算公式的函数规则 
* 输出 感知器的输出计算公式是：$$y=f(w*x+b)$$


**单独的感知器能够拟合任意的线性函数，因此用于线性分类和线性回归等问题。**

感知器的计算公式重除了x是输入，f是激活函数，参数w、b都是**如何获得**去参与计算的呢？

##引出感知器规则算法：
权重项和偏置项的值最终由训练集的数据迭代计算得到
**初始值:** 0
**迭代次数:** 人为设置的训练次数
**计算公式：** $$wi\leftarrow wi+\Delta wi
$$$$b\leftarrow b+\Delta b
$$其中：
$$\Delta wi = \eta (t-y)xi
$$这里的t是对应的真实值-期待的正确值，y是输出计算公式计算的值
$$
\Delta b = \eta (t-y)x0
$$

##编程实战：实现感知器

address: *Demo/Perception*

