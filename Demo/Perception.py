from functools import reduce
class Perception(object):
    # input_num代表对应输入\输出样本的向量维度
    # 初始化感知器,目的是训练weights和bias
    def __init__(self,input_num,activator):
        self.activator=activator
        self.weights=[0.0 for _ in range(input_num)]
        self.bias=0.0
        self.x0=1
    
    # 打印训练后学习到的权重和偏置
    def __str__(self):
        return 'weights:\t%s\nbias:\t%f\n' %(self.weights,self.bias)
    
    def predict(self,input_vec):
        # zip将对应的权重和样本特征打包
        # input_vec样本输入是[x1,x2,x3,...xm]
        # weights对应样本特征个数[w1,w2,w3,...wm]
        # zip(input,weights)结果是[(x1,w1),(x2,w2),...(xm,wm)]
        # 对应相乘 xi*wi map函数
        # map(function,iterable)结果是[x1*w1,x2*w2,...xm*wm]
        # reduce函数累加 redece(function,iterable)结果是x1*w1+x2*w2+...xm*wm
        # reduce的整个完整过程是w*x,
        # return的完整数学计算公式是f(w*x+b)
        return self.activator(
            reduce(lambda a,b:a+b,
            map(lambda elements:elements[0]*elements[1],
            zip(input_vec,self.weights)),
            0.0)+self.bias*self.x0)
    
    
    '''
    训练感知器
    '''
    #感知器训练函数 
    def train(self,input_vecs,labels,rate,iteration):
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)
    
    # 全部样本的一次周期训练
    def _one_iteration(self,input_vecs,labels,rate):
        samples=zip(input_vecs,labels)
        for (input_vec,label) in samples:
            # 抽出的每个样本,目的是为了训练权重和偏置
            output=self.predict(input_vec)
            self._weights_update(rate,label,output,input_vec)
    
    '''
    感知器规则
    '''
    # feed sample to update weights  
    def _weights_update(self,rate,label,output,input_vec):
        delta = label-output
        self.weights=list(map(lambda elements:rate*delta*elements[0]+elements[1],
        zip(input_vec,self.weights)))
        self.bias+=rate*delta*self.x0

# 既定的激活函数
def f(x):
    return 1 if x>0 else 0

# 训练集
# 样本集------真实结果集
def get_training_dataset():
    # [1,1]->1,[1,0]->0,[0,1]->0,[0,0]->0
    input_vecs=[[1,1],[1,0],[0,1],[0,0]]
    labels=[1,0,0,0]
    return input_vecs,labels

def train_and_perception():
    # 面向函数编程的创建对象方法
    p = Perception(2,f)
    input_vecs,labels=get_training_dataset()
    p.train(input_vecs,labels,0.1,10)
    return p

if __name__=='__main__':
    and_percrption=train_and_perception()
    print(and_percrption)
    # 测试的结果和数据的关联性很大
    # 权重和偏置的计算已经拟合 随着训练次数的增加朝一个确定值方向发展   
    print('1 and 1 = %d' %and_percrption.predict([1,1]))
    print('1 and 0 = %d' %and_percrption.predict([1,0]))
    # print('1 and 1 = %d' %and_percrption.predict([3,2]))
    print('0 and 0 = %d' %and_percrption.predict([0,0]))
    print('0 and 1 = %d' %and_percrption.predict([0,1]))
