from functools import reduce
class LinearUnit(object):
    def __init__(self,input_num,activator):
        self.activator=activator
        self.weights=[0.0 for i in range(input_num)]
        self.bias=0.0
        self.x0=1
    # 检验用于测试集训练的参数
    def __str__(self):
        return 'weights:%s\nb:\t%f\n' %(self.weights,self.bias)
    
    '''
    线性单元模型，训练好模型的预测值。
    '''
    def predict(self,input_vec):
        return self.activator(
            reduce(lambda a,b:a+b,
            map(lambda elements:elements[0]*elements[1],
            zip(input_vec,self.weights)),0.0)+self.x0*self.bias)
    
    '''
    训练过程，迭代的次数
    '''
    def train(self,input_vecs,labels,rate,iteration):
        for iter in range(iteration):
            self.one_iteration(input_vecs,labels,rate)
    
    '''
    每次迭代的具体细节
    '''
    def one_iteration(self,input_vecs,labels,rate):
        samples=zip(labels,input_vecs)
        for (label,input_vec) in samples:
            output=self.predict(input_vec)
            self.update_weights(rate,label,output,input_vec)
    
    '''
    更新步调W_new,优化算法接近于0, 
    得到的模型使得目标函数最小
    '''
    def update_weights(self,rate,label,output,input_vec):
        delta=label-output
        self.weights=list(map(lambda elements:elements[1]+rate*delta*elements[0],
        zip(input_vec,self.weights)))
        self.bias+=rate*delta*self.x0

'''
线性单元激活函数，返回的是实际值，而非0、1的分类
'''
def f(x):
    return x

'''
训练集的数据
'''
def get_training_dataset():
    input_vecs=[[5],[3],[8],[1.4],[10.1]]

    labels=[5500,2300,7600,1800,11400]
    return input_vecs,labels

'''
训练集的训练过程
使用数据训练线性单元
'''
def train_and_predict():
    lu=LinearUnit(1,f)
    input_vecs,labels=get_training_dataset()
    lu.train(input_vecs,labels,0.01,10)
    return lu

if __name__=="__main__":
    # 得到线性单元
    linearUnit=train_and_predict()
    print(linearUnit)

    print('Work 3.4 years,monthly salary =%.2f' %linearUnit.predict([3.4]))
    print('Work 15 years,monthly salary =%.2f' %linearUnit.predict([15]))
    print('Work 1.5 years,monthly salary =%.2f' %linearUnit.predict([1.5]))
    print('Work 6.3 years,monthly salary =%.2f' %linearUnit.predict([6.3]))
