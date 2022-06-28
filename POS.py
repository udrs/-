import numpy as np
import matplotlib.pyplot as plt
import utils

POP_SIZE = 100 #群の個体数
N_GENERATIONS = 100 #世代数
C1 = 0.2 #各探索個体の最良位置の学習係数
C2 = 10 #個体群全体での最良値の学習係数
W = 0.6 #慣性定数
X_MIN = 0 #解答空間の最小値
X_MAX = 15 #解答空間の最大値
BEST_ANSWERS = [] #最適な結果を記録
LIMIT_SPEED_INITAIL = 10
LIMIT_SPEED_MIDDLE = 5
LIMIT_SPEED_FINALLY = 1

class utils(object):
    def __init__(self, population_size=POP_SIZE, max_steps=N_GENERATIONS):
        self.population_size=population_size

    

    def clamp(n, minn, maxn):
        n = np.maximum(n, minn)
        n = np.minimum(n, maxn)
        return n

    def limit_speed(v, time): # 模倣焼きなまし法，初期段階では非常に低い速度制
    #限、中期段階では中程度の速度制限、後期段階では高い速度制限
    #print("time=",time)
        if(time < N_GENERATIONS*0.3): # 初期段階
            speedlimit_up = [[LIMIT_SPEED_INITAIL]*3]*POP_SIZE
            speedlimit_down = [[LIMIT_SPEED_INITAIL*(-1)]*3]*POP_SIZE
            v = np.minimum(v, speedlimit_up)
            v = np.maximum(v, speedlimit_down)
        elif(time < N_GENERATIONS*0.7): # 中期段階
            speedlimit_up = [[LIMIT_SPEED_MIDDLE]*3]*POP_SIZE
            speedlimit_down = [[LIMIT_SPEED_MIDDLE*(-1)]*3]*POP_SIZE
            v = np.minimum(v, speedlimit_up)
            v = np.maximum(v, speedlimit_down)
        else: # 後期段階
            speedlimit_up = [[LIMIT_SPEED_FINALLY]*3]*POP_SIZE
            speedlimit_down = [[LIMIT_SPEED_FINALLY*(-1)]*3]*POP_SIZE
            v = np.minimum(v, speedlimit_up)
            v = np.maximum(v, speedlimit_down)
        return v
        
    def F(x1, x2, x3):
	    return 2*x1**2-3*x2**2-4*x1+5*x2+x3


class PSO(object):
    def __init__(self, population_size=POP_SIZE, max_steps=N_GENERATIONS):
        self.population_size = population_size  # 粒子群数量
        self.dim = 3  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [X_MIN, X_MAX]  # 解空间范围 
        self.x = np.random.randint(self.x_bound[0],self.x_bound[1]+1, size=(POP_SIZE, self.dim)) #每一行就是一个3维度的向量，对应上x1，x2，x3
        self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速度
        fitness = self.calculate_fitness(self.x)
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmax(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.max(fitness)  # 全局最佳适应度
        BEST_ANSWERS.append(self.global_best_fitness)

    def calculate_fitness(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2] 
        pred = utils.F(x1, x2, x3)
        return pred

    def evolve(self):
        for _ in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
# 更新速度と重み付け
            self.v = W*self.v+C1*r1*(self.p-self.x)+C2*r2*(self.pg-self.x) # 個体の速度を計算する
            self.v = utils.limit_speed(self.v, _) # 段階的速度制限
            minn = [[self.x_bound[0]]*self.dim]*self.population_size
            maxn = [[self.x_bound[1]]*self.dim]*self.population_size
            if _ == 0:
                fitness = self.calculate_fitness(self.x)
            best_fitness_place_value = self.x[np.argmax(fitness)] # 最良位置の値
            best_fitness_place = np.argmax(fitness) # 最良位置のナンバリング
            self.x = utils.clamp(self.v + self.x, minn, maxn) # 粒子の位置を更新する
            self.x[best_fitness_place] = best_fitness_place_value # 保留最优解
            fitness = self.calculate_fitness(self.x)
# 個体の更新
            update_id = np.greater(self.individual_best_fitness, fitness)

            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id] 
            # 個体群全体での最良値の更新
            if np.max(fitness) > self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            BEST_ANSWERS.append(np.min(fitness))
 
pso = PSO()
pso.evolve()
print("Result:",min(BEST_ANSWERS))
plt.plot(BEST_ANSWERS)
plt.show()
