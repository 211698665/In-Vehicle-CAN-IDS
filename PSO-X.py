import os
import pickle
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier as xgbc
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
import random
import matplotlib.pyplot as plt

from tqdm import tqdm


class PSO(object):
    def __init__(self, particle_num, particle_dim, iter_num, c1, c2, w, max_value, min_value):
        '''参数初始化
        particle_num(int):粒子群的粒子数量
        particle_dim(int):粒子维度，对应待寻优参数的个数
        iter_num(int):最大迭代次数
        c1(float):局部学习因子，表示粒子移动到该粒子历史最优位置(pbest)的加速项的权重
        c2(float):全局学习因子，表示粒子移动到所有粒子最优位置(gbest)的加速项的权重
        w(float):惯性因子，表示粒子之前运动方向在本次方向上的惯性
        max_value(float):参数的最大值
        min_value(float):参数的最小值
        '''
        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.iter_num = iter_num
        self.c1 = c1  ##通常设为2.0
        self.c2 = c2  ##通常设为2.0
        self.w = w
        self.max_value = max_value
        self.min_value = min_value

    def swarm_origin(self):
        particle_loc = []
        particle_dir = []
        for i in range(self.particle_num):
            tmp1 = []
            tmp2 = []
            for j in range(self.particle_dim):
                a = random.random()
                b = random.random()
                tmp1.append(a * (self.max_value - self.min_value) + self.min_value)
                tmp2.append(b)
            particle_loc.append(tmp1)
            particle_dir.append(tmp2)

        return particle_loc, particle_dir

    def fitness(self, particle_loc):
        fitness_value = []
        for i in range(self.particle_num):
            # model = LGBMClassifier(random_state=1, n_jobs=-1, n_estimators=int(particle_loc[i][0]),
            #                        max_depth=int(particle_loc[i][1]), learning_rate=int(particle_loc[i][2]))
            model = xgbc(random_state=1, n_jobs=-1, n_estimators=int(particle_loc[i][0]),max_depth=int(particle_loc[i][1]), learning_rate=int(particle_loc[i][2]))
            # model = CatBoostClassifier(random_state=1, n_jobs=-1, iterations=int(particle_loc[i][0]),
            #                             depth=int(particle_loc[i][1]), learning_rate=int(particle_loc[i][2]))
            #model=MLPClassifier(hidden_layer_sizes=(100,),learning_rate=int(particle_loc[i][0]))
            cv_scores = model_selection.cross_val_score(model, trainX, trainY, cv=3, scoring='accuracy')
            fitness_value.append(cv_scores.mean())
        current_fitness = 0.0
        current_parameter = []
        for i in range(self.particle_num):
            if current_fitness < fitness_value[i]:
                current_fitness = fitness_value[i]
                current_parameter = particle_loc[i]

        return fitness_value, current_fitness, current_parameter

    def update(self, particle_loc, particle_dir, gbest_parameter, pbest_parameters):
        for i in range(self.particle_num):
            a1 = [x * self.w for x in particle_dir[i]]
            a2 = [y * self.c1 * random.random() for y in
                  list(np.array(pbest_parameters[i]) - np.array(particle_loc[i]))]
            a3 = [z * self.c2 * random.random() for z in list(np.array(gbest_parameter) - np.array(particle_dir[i]))]
            particle_dir[i] = list(np.array(a1) + np.array(a2) + np.array(a3))
            particle_loc[i] = list(np.array(particle_loc[i]) + np.array(particle_dir[i]))

        parameter_list = []
        for i in range(self.particle_dim):
            tmp1 = []
            for j in range(self.particle_num):
                tmp1.append(particle_loc[j][i])
            parameter_list.append(tmp1)
        value = []
        for i in range(self.particle_dim):
            tmp2 = []
            tmp2.append(max(parameter_list[i]))
            tmp2.append(min(parameter_list[i]))
            value.append(tmp2)

        for i in range(self.particle_num):
            for j in range(self.particle_dim):
                particle_loc[i][j] = (particle_loc[i][j] - value[j][1]) / (value[j][0] - value[j][1]) * (
                        self.max_value - self.min_value) + self.min_value

        return particle_loc, particle_dir

    def plot(self, results):
        X = []
        Y = []
        for i in range(self.iter_num):
            X.append(i + 1)
            Y.append(results[i])
        plt.plot(X, Y)
        plt.xlabel('Number of iteration', size=15)
        plt.ylabel('Value of CV', size=15)
        plt.title('PSO parameter optimization')
        plt.show()

    def main(self):
        results = []
        log = []
        best_fitness = 0.0
        particle_loc, particle_dir = self.swarm_origin()
        gbest_parameter = []
        for i in range(self.particle_dim):
            gbest_parameter.append(0.0)
        pbest_parameters = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                tmp1.append(0.0)
            pbest_parameters.append(tmp1)
        fitness_value = []
        for i in range(self.particle_num):
            fitness_value.append(0.0)

        for i in tqdm(range(self.iter_num)):
            current_fitness_value, current_best_fitness, current_best_parameter = self.fitness(particle_loc)
            for j in range(self.particle_num):
                if current_fitness_value[j] > fitness_value[j]:
                    pbest_parameters[j] = particle_loc[j]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                gbest_parameter = current_best_parameter

            print('iteration is :', i + 1, ';Best parameters:', gbest_parameter, ';Best fitness', best_fitness)
            results.append(best_fitness)
            fitness_value = current_fitness_value
            particle_loc, particle_dir = self.update(particle_loc, particle_dir, gbest_parameter, pbest_parameters)
            log.append((i, gbest_parameter[0], gbest_parameter[1], best_fitness))

            print(particle_loc)
            os.makedirs('fig', exist_ok=True)
            plt.plot(list(range(0, len(current_fitness_value))), current_fitness_value)
            plt.title(i)
            plt.savefig(f"fig/{i}_适应度_{current_best_fitness}.png")
            plt.show()
            a = np.array(particle_loc)
            x = a[:, 0].tolist()
            y = a[:, 1].tolist()

            plt.scatter(x, y)
            plt.title(i)
            plt.savefig(f"fig/{i}_粒子位置_{current_best_fitness}.png")
            plt.show()

        results.sort()
        self.plot(results)
        print('Final parameters are :', gbest_parameter)
        with open("log.pkl", "wb") as f:
            pickle.dump(log, f)


if __name__ == '__main__':
    cwd = os.getcwd()
    train = cwd + "/train_data.csv"
    train_data = pd.read_csv(train)
    trainY = train_data['Lable']
    trainX = train_data.drop(columns='Lable')
    particle_num = 100  # 粒子群的粒子数量
    particle_dim = 3  # 调优参数个数
    iter_num = 100  # 迭代次数
    c1 = 2  # 局部学习因子
    c2 = 2  # 全局学习因子
    w = 0.8  # 惯性因子，表示粒子之前运动方向在本次方向上的惯性
    max_value = 200  # 根据参数情况调整
    min_value = 0  # 根据参数情况调整
    pso = PSO(particle_num, particle_dim, iter_num, c1, c2, w, max_value, min_value)
    pso.main()
