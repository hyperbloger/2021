from functools import reduce
# from functools import map
# from functools import zip
class Perception(object):
    def __init__(self,input_num,activator):
        self.activator=activator;
        # 权重项和偏置项初始化
        # 权重项对应样本输入的向量维度
        self.weights=[0.0 for i in range(input_num)]

        #偏置项是b,感知器算法中是常量
        self.bias=0.0
        print(type(self.bias))
        self.x0=1
    # 打印函数.方便我们检查每次迭代的权重项和偏置项
    def __str__(self):
        return 'count:\t%d\nweights\t:%s\nbias\t:%f\n' % (self.count,self.weights,self.bias)
    # 感知器输出计算公式
    def predict(self,input_vec):
        return self.activator(
            # zip函数将列表的元素按照最短列表长度的打包
            #input_vec[x1,x2,...xn]和self.weights[w1,w2,...wn]打包
            # 组成[(x1,w1),(x2,w2),...(wn,xn)]元组的列表
            # map函数将列表中的数按照规则计算
            # 结果是[x1*w1,x2*w2,...xn*wn]
            # 用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，
            # 得到的结果再与第三个数据用 function 函数运算
            reduce(lambda a,b:a+b,
                map(lambda elements:elements[0]*elements[1],
                    zip(input_vec,self.weights)),0.0)+self.bias*self.x0)
            
    #感知器训练算法-得到训练后的w和b
    # input_vecs是样本的全部[样本1,样本2,样本n];一个样本是一个input_vec
    # labels是真实值,对应训练样本正确的输出[样本1正确结果,样本2正确结果,...,样本nz正确结果]
    # rate是学习率,常量参数,控制迭代的学习速率
    # iteration是训练的轮数;训练轮数是人为的,完整的样本处理是一轮
    def train(self,input_vecs,labels,rate,iteration):
        # 一个疑问 rate*iteration==1
        for i in range(iteration):
            self.one_iteration(input_vecs,labels,rate)

    def one_iteration(self,input_vecs,labels,rate):
        '''
        一次完整的迭代过程,具体delta w和delta b的计算过程:训练
        '''
        # 完成对应工作
        samples=zip(input_vecs,labels)    
        for input_vec,label in samples:
            self.count=self.count+1
            output=self.predict(input_vec)
            self.__update_weight(input_vec,label,output,rate)
    
    def __update_weight(self,input_vec,label,output,rate):
        delta=label-output
        self.weights=map(
            lambda elements:rate*delta*elements[0]+elements[1],
            zip(input_vec,self.weights))
        # print(list(self.weights))
        self.bias=self.bias+rate*delta*self.x0
        # print(self.bias)

def f(x):
    '''
    定义激活函数
    '''    
    return 1 if x>0 else 0

def get_training_dataset():
    '''
    基于and真值表构建训练数据
    '''
    # 
    input_vecs=[[1,1],[0,0],[1,0],[0,1]]
    # 
    labels=[1,0,0,0]
    return input_vecs,labels

def training_and_perception():
    '''
    训练感知器
    '''
    p=Perception(2,f)
    input_vecs,labels=get_training_dataset()
    p.train(input_vecs,labels,0.1,10)
    return p

if __name__=='__main__':
    and_perception = training_and_perception()
    print(and_perception)
    print('1 and 1=  %d' %and_perception.predict([1,1]))
    print('1 and 0=  %d' %and_perception.predict([1,0]))
    print('0 and 1=  %d' %and_perception.predict([0,1]))
    print('0 and 0=  %d' %and_perception.predict([0,0]))


