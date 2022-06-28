
import matplotlib.pyplot as plt
import numpy as np

DNA_SIZE = 4  #基因长度
POP_SIZE = 100  #种群总个数
CROSSOVER_RATE = 0.8 #交叉概率
MUTATION_RATE = 0.1 #变异概率
N_GENERATIONS = 100  #迭代次数
best_answer = []


class Genetic_ALG(object):
    def __init__(self, POP_SIZE=POP_SIZE, DAN_SIZE=DNA_SIZE, CROSSOVER_RATE=CROSSOVER_RATE,MUTATION_RATE=MUTATION_RATE):
        self.dim = 3  # 搜索空间的维度
        self.DNA_SIZE=DAN_SIZE
        self.POP_SIZE=POP_SIZE
        self.CROSSOVER_RATE=CROSSOVER_RATE
        self.MUTATION_RATE=MUTATION_RATE
    
    def Process(self,pop):
        fitness = self.get_fitness(pop)
        for _ in range(N_GENERATIONS): #迭代N代
            pop = np.array(self.crossover_and_mutation(pop))
            fitness = self.get_fitness(pop)
            pop = self.select(pop, fitness) #选择生成新的种群
        print(max(best_answer))

    def translateDNA(self,pop):
        x1_pop = pop[:,0:4]  
        x2_pop = pop[:,4:8] 
        x3_pop = pop[:,8:12] 
        x1 = x1_pop.dot(2**np.arange(self.DNA_SIZE))
        x2 = x2_pop.dot(2**np.arange(self.DNA_SIZE))
        x3 = x3_pop.dot(2**np.arange(self.DNA_SIZE))
        return x1,x2,x3

    def crossover_and_mutation(self,pop):
        new_pop = []
        for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
            child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if np.random.rand() < self.CROSSOVER_RATE:			#产生子代时概率发生交叉
                mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
                cross_points = np.random.randint(low=0, high=DNA_SIZE*3)	#随机产生交叉的点
                child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
            self.mutation(child)	#每个后代有一定的机率发生变异
            new_pop.append(child)
        return new_pop

    def mutation(self,child):
        if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, DNA_SIZE)	#随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point]^1 	#将变异点的二进制为反转

    def get_fitness(self,pop):    #用于计算适合度矩阵
        x1,x2,x3 = self.translateDNA(pop)
        pred = self.Fun(x1, x2, x3)
        best_answer.append(pred.max())
        return (pred - np.min(pred)) 

    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=(fitness)/(fitness.sum())) 
        return pop[idx] 
    
    def Fun(self,x1, x2, x3):
        return 2*x1**2-3*x2**2-4*x1+5*x2+x3



if __name__ == "__main__":
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*3))
    AG=Genetic_ALG()
    a=AG.Process(pop)

    #绘制最优解
    plt.plot(best_answer)
    plt.show()


    