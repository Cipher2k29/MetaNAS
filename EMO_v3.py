from email.policy import default
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import collections, argparse, time, logging, sys
import matplotlib.pyplot as plt

from EMO_public import P_generator, NDsort, F_distance, F_mating, F_EnvironmentSelect

from model_training import solution_evaluation
from utils import dagnode, create__dir, Plot_network
from Node import Operations_11_name, NetworkCIFAR

from Build_Dataset import build_search_cifar10, build_search_Optimizer_Loss, build_search_cifar100
from torch.utils.tensorboard import SummaryWriter


class individual():
    def __init__(self, dec):
        # dec
        # dag
        # num_node
        self.dec = dec
        self.re_duplicate()
        # self.trans2bin()# if dec is (int10,op)
        self.trans2dag()

    # def trans2bin(self):
    #     self.bin_dec = []
    #     self.conv_bin_dec = []
    #     self.redu_bin_dec =[]
    #
    #     for i in range(2):
    #         temp_dec = []
    #         for j in range(int(len(self.dec[i])/2)):
    #             bin_value = bin(self.dec[i][2*j])
    #             temp_list = [int(i) for i in bin_value[2:] ]
    #             if len(temp_list)<j+2:
    #                 A = [0]*(j+2 - len(temp_list))
    #                 A.extend(temp_list)
    #                 temp_list = A.copy()
    #             temp_list.extend([self.dec[i][2*j+1]])
    #             temp_dec.append(temp_list)
    #         self.bin_dec.append(temp_dec)
    #
    #     temp = [self.conv_bin_dec.extend(i) for i in self.bin_dec[0]]
    #     del temp
    #     temp = [self.redu_bin_dec.extend(i) for i in self.bin_dec[1]]
    #     del temp
    def re_duplicate(self):
        # used for deleting the nodes not actived

        for i, cell_dag in enumerate(self.dec):
            L = 0
            j = 0
            zero_index = []
            temp_dec = []
            while L < len(cell_dag):
                S = L
                L += 3 + j
                node_j_A = np.array(cell_dag[S:L]).copy()
                node_j = node_j_A[:-1]
                if node_j.sum() - node_j[zero_index].sum() == 0:
                    zero_index.extend([j + 2])
                else:
                    temp_dec.extend(np.delete(node_j_A, zero_index))
                j += 1

            self.dec[i] = temp_dec.copy()

    def trans2dag(self):
        self.dag = []
        self.num_node = []

        for i in range(2):
            dag = collections.defaultdict(list)
            dag[-1] = dagnode(-1, [], None)
            dag[0] = dagnode(0, [0], None)

            j = 0
            L = 0
            while L < len(self.dec[i]):
                S = L
                L += 3 + j
                node_j = self.dec[i][S:L]
                dag[j + 1] = dagnode(j + 1, node_j[:-1], node_j[-1])
                j += 1
            self.num_node.extend([j])
            self.dag.append(dag)
            del dag

    def evaluate(self, train_queue, valid_queue, args):

        self.fitness = np.random.rand(4, )

        model = NetworkCIFAR(args, args.classes, args.search_layers, args.search_channels, self.dag, args.search_use_aux_head,
                             args.search_keep_prob, args.search_steps, args.search_drop_path_keep_prob,
                             args.search_channels_double)

        self.fitness = solution_evaluation(model, train_queue, valid_queue, args)

        del model


class EMO():
    def __init__(self, args, visualization=False):  # [5,8]
        self.args = args
        self.popsize = args.popsize
        self.Max_Gen = args.Max_Gen
        self.Gen = 0
        self.initial_range_node = args.range_node
        self.save_dir = args.save

        self.get_op_index()
        self.op_num = len(Operations_11_name)
        self.max_length = self.op_index[-1] + 1
        self.coding = 'Binary'

        self.visualization = visualization

        self.Population = []
        self.Pop_fitness = []
        self.finess_best = 0.0 #230919 0 -> 0.0

        self.offspring = []
        self.off_fitness = []

        self.tour_index = []
        self.FrontValue = []
        self.CrowdDistance = []
        self.select_index = []

        self.build_dataset()

        self.threshold = 0.08  # 0.08

        self.MatingPool = []
        self.top1_err_pred = []

    def get_op_index(self):
        self.op_index = []
        L = 0
        for i in range(self.initial_range_node[1]):
            L += 3 + i
            self.op_index.extend([L - 1])

    def build_dataset(self):
        if args.classes == 10 :
            train_queue, valid_queue = build_search_cifar10(args=self.args, ratio=0.9,
                                                         num_workers=self.args.search_num_work)  # change the used dataset
        elif args.classes == 100:
            train_queue, valid_queue = build_search_cifar100(args=self.args, ratio=0.9,
                                                         num_workers=self.args.search_num_work)  # change the used dataset
        self.train_queue = train_queue
        self.valid_queue = valid_queue

    def initialization(self):
        for i in range(self.popsize):
            rate = (i + 1) / self.popsize  # used for controlling the network structure between 'line' and 'Inception'
            node_ = np.random.randint(self.initial_range_node[0], self.initial_range_node[1] + 1, 2)

            list_individual = []

            for i, num in enumerate(node_):
                op = np.random.randint(0, self.op_num, num)  # 12 Operation_11(SELayer), 7 Operation_7
                if i == 0:
                    op_c = np.random.randint(0, 4, num)  # Conv index [0 1 2 3] in Operation_11, Operation_7
                else:
                    op_c = np.random.randint(4, 10, num)  # Pool index[4 5 6 7 8 9] in Operation_11, [4 5] Operations_7
                in_dicator = np.random.rand(num, ) < 0.8  # 0.8
                op[in_dicator] = op_c[in_dicator]

                L = 2
                dag_list = []
                for j in range(num):
                    L += 1
                    link = np.random.rand(L - 1)
                    link[-1] = link[-1] > rate
                    link[0:2] = link[0:2] < rate
                    link[2:-1] = link[2:-1] < 2 / len(link[2:-1]) if len(link[2:-1]) != 0 else []  # 2

                    if link.sum() == 0:
                        if rate < 0.5:
                            link[-1] = 1
                        else:
                            if np.random.rand(1) < 0.5:
                                link[1] = 1
                            else:
                                link[0] = 1

                    link = np.int64(link)
                    link = link.tolist()
                    link.extend([op[j]])
                    dag_list.extend(link)
                list_individual.append(dag_list)

            self.Population.append(individual(list_individual))

        Up_boundary = np.ones((self.max_length))
        Up_boundary[self.op_index] = self.op_num - 1
        Low_boundary = np.zeros((self.max_length))

        self.Boundary = np.vstack((Up_boundary, Low_boundary))
        self.Pop_fitness = self.evaluation(self.Population)

        self.finess_best = np.min(self.Pop_fitness[:, 0])  # get the min in colunm 0
        self.save('initial')

    def save(self, path=None):

        if path is None:
            path = 'Gene_{}'.format(self.Gen + 1)
        whole_path = '{}/{}/'.format(self.save_dir, path)
        create__dir(whole_path)

        fitness_file = whole_path + 'fitness.txt'
        np.savetxt(fitness_file, self.Pop_fitness, delimiter=' ')

        Pop_file = whole_path + 'Population.txt'
        with open(Pop_file, "w") as file:
            for j, solution in enumerate(self.Population):
                file.write('solution {}: {} \n'.format(j, solution.dec))    # 0 ~ popsize-1 (*** 1 ~ popsize, in logging )

        best_index = np.argmin(self.Pop_fitness[:, 0])                      # 0 ~ popsize-1
        solution = self.Population[best_index]
        Plot_network(solution.dag[0], '{}/{}_conv_dag.png'.format(whole_path, best_index))
        Plot_network(solution.dag[1], '{}/{}_reduc_dag.png'.format(whole_path, best_index))

    def evaluation(self, Pop):
        # 是否 normalize fitness
        # return np.random.rand(len(Pop),2)
        fitness = np.zeros((len(Pop), 4))
        dec_array = []
        dec_fitness = []
        for i, solution in enumerate(Pop):
            logging.info('solution: {0:>2d}'.format(i + 1))
            print('solution: {0:>2d}'.format(i + 1))
            solution.evaluate(self.train_queue, self.valid_queue, self.args)
            solution.fitness = (solution.fitness[0].cpu(),) + solution.fitness[1:] 
            fitness[i] = solution.fitness
            print('fitness:', fitness[i])  # ,'solution.fitness',solution.fitness)

            #### 0716 surrogate ####
            from acc_predictor.factory import get_acc_predictor
            import pandas as pd
            # print('solution.dec:',solution.dec,'\n', 'solution.fitness[0]:',solution.fitness[0])

            dec_array.append(solution.dec[0])
            dec_fitness.append(solution.fitness[0])

        # print(dec_array)

        if args.is_surrogate == True:
            dectemp = []
            dectemp = pd.concat([pd.DataFrame({'{}'.format(index): labels}) for index, labels in enumerate(dec_array)],
                                axis=1)
            dectemp = dectemp.fillna(0).values.T
            dec_array = np.array(dectemp)
            dec_array = np.array(dectemp)
            print('dec_array', dec_array, 'shape[1]', dec_array.shape[1], 'shape:', dec_array.shape)
            print('fitness', dec_fitness)
            
            dec_array = dec_array[:, :9]  
            print('testing', 'dec_array', dec_array, 'shape[1]', dec_array.shape[1], 'shape:', dec_array.shape)

            acc_predictor = get_acc_predictor('rbf', dec_array, dec_fitness)  ##
            # global top1_err_pred
            self.top1_err_pred = acc_predictor.predict(dec_array)  ## this two lines euqal to msunas from nsganetv2
            print('acc_predictor_winner:', acc_predictor.name, '\n', 'top1_err_pred:',
                  self.top1_err_pred)  # if surrogate 'as', acc_predictor.winner
            logging.info('acc_predictor_winner: {}'.format(acc_predictor.name))
            logging.info('top1_err_pred of each solution: \n{}'.format(self.top1_err_pred))

            torch.save(acc_predictor, 'acc_predictor.pt')

        return fitness[:, :2] 

    def Binary_Envirmental_tour_selection(self):
        self.MatingPool, self.tour_index = F_mating.F_mating(self.Population.copy(), self.FrontValue,
                                                             self.CrowdDistance)

    def genetic_operation(self):
        offspring_dec = P_generator.P_generator(self.MatingPool, self.Boundary, self.coding, self.popsize,
                                                self.op_index)
        offspring_dec = self.deduplication(offspring_dec)
        self.offspring = [individual(i) for i in offspring_dec]
        self.off_fitness = self.evaluation(self.offspring)

    def first_selection(self):
        Population = []
        Population.extend(self.Population)
        Population.extend(self.offspring)

        Population_temp = []
        for i, solution in enumerate(Population):
            if solution.fitness[0] < self.finess_best + self.threshold:
                Population_temp.append(solution)

        FunctionValue = np.zeros((len(Population_temp), 2))
        for i, solution in enumerate(Population_temp):
            FunctionValue[i] = solution.fitness[:2]

        return Population_temp, FunctionValue

    def Envirment_Selection(self):

        # Population = []
        # Population.extend(self.Population)
        # Population.extend(self.offspring)
        # FunctionValue = np.vstack((self.Pop_fitness, self.off_fitness))

        Population, FunctionValue = self.first_selection()

        Population, FunctionValue, FrontValue, CrowdDistance, select_index = F_EnvironmentSelect. \
            F_EnvironmentSelect(Population, FunctionValue, self.popsize)

        self.Population = Population
        self.Pop_fitness = FunctionValue
        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance
        self.select_index = select_index

        self.finess_best = np.min(self.Pop_fitness[:, 0])

    def deduplication(self, offspring_dec):
        pop_dec = [i.dec for i in self.Population]
        dedup_offspring_dec = []
        for i in offspring_dec:
            if i not in dedup_offspring_dec and i not in pop_dec:
                dedup_offspring_dec.append(i)

        return dedup_offspring_dec

    def print_logs(self, since_time=None, initial=False):
        if initial:

            logging.info(
                '********************************************************************Initializing**********************************************')
            print(
                '********************************************************************Initializing**********************************************')
        else:
            used_time = (time.time() - since_time) / 60

            logging.info(
                '*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                '*****************************************'.format(self.Gen + 1, self.Max_Gen, used_time))

            print(
                '*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                '*****************************************'.format(self.Gen + 1, self.Max_Gen, used_time))

    def plot_fitness(self):
        if self.visualization:
            plt.clf()
            plt.scatter(self.Pop_fitness[:, 0], self.Pop_fitness[:, 1])
            plt.xlabel('Error')
            plt.ylabel('parameters: MB')
            plt.pause(0.001)

    def Main_loop(self):
        since_time = time.time()
        plt.ion()

        self.print_logs(initial=True)
        self.initialization()
        self.plot_fitness()

        self.FrontValue = NDsort.NDSort(self.Pop_fitness, self.popsize)[0]
        self.CrowdDistance = F_distance.F_distance(self.Pop_fitness, self.FrontValue)
        
        self.Binary_Envirmental_tour_selection()
        print('First MatingPool & tour_index:', self.MatingPool, self.tour_index)
        self.genetic_operation()
        self.Envirment_Selection()
        #used in cifar10. 

        while self.Gen < self.Max_Gen:
            if (args.is_surrogate == True) and (args.k > 0) and (min(np.array(self.top1_err_pred)[:, 0]) > 0.168):  
                logging.info('****extra generation {}****:'.format(args.k))
                print('****Extra Gene****')
                args.k -= 1
                self.print_logs(since_time=time.time())

                # self.Binary_Envirmental_tour_selection()
                self.MatingPool, self.tour_index = F_mating.F_mating(self.Population.copy(), self.FrontValue,
                                                                        self.CrowdDistance)

                print('MatingPool & tour_index:', self.MatingPool, self.tour_index)

                self.genetic_operation()
                self.Envirment_Selection()

                # self.plot_fitness()
                # self.save()

            else:
                if args.is_surrogate == True:
                    if min(np.array(self.top1_err_pred)[:, 0]) <= 0.168:
                        logging.info('**** Goal Achieved ****')
                        print('**** Goal Achieved ****')
                    else:
                        logging.info('**** Failed; Print Last Gene ****')
                        print('**** Failed; Print Last Gene ****')
                args.is_surrogate = False
                args.search_epochs = 110  # 110
                args.search_steps = int(np.ceil(45000 / args.search_train_batch_size)) * args.search_epochs
                self.print_logs(since_time=since_time)

                print('whether MatingPool & tour_index changed or not', self.MatingPool, self.tour_index)

                self.genetic_operation()
                self.Envirment_Selection()

                self.plot_fitness()
                self.save()
                self.Gen += 1

                args.is_surrogate = True
                args.search_epochs = 35
                args.search_steps = int(np.ceil(45000 / args.search_train_batch_size)) * args.search_epochs
                args.k = 1 # need adjust
                # break

        plt.ioff()
        plt.savefig("{}/final.png".format(self.save_dir))


if __name__ == "__main__":

    # ===================================  args  ===================================
    # ***************************  common setting******************
    parser = argparse.ArgumentParser(description='test argument')
    parser.add_argument('--seed', type=int, default=1000) #1000 if seed 3407, error in surrogate
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-save', type=str, default='result')
    # ***************************  EMO setting******************
    parser.add_argument('-range_node', type=list, default=[5, 12])  # [5,12]
    parser.add_argument('-popsize', type=int, default=20)  # 10 # 20	# the number of solutions in one gene
    parser.add_argument('-Max_Gen', type=int, default=1)  # 5 # 25 # the number of Final Genes ## 240408: CIFAR10 for multiple loops, CIFAR100 can only set to 1 with small epochs, or boardcast error, but with the period mutation, set to 1 is enough, AND We just need a shell to make loops for multiple initialization.

    # *** meta-lr setting ***
    parser.add_argument('-classes', type=int, default=10)  # need adjust, 10 in cifar10, 100 in cifar100
    parser.add_argument('-is_surrogate', action='store_true', default=True)  #
    parser.add_argument('-k', type=int, default=3)  #3 # maxtime of re-init

    # ***************************  dataset setting******************
    parser.add_argument('-data', type=str, default="~/storage/work_dir/lgl/data/cifar10")
    parser.add_argument('-search_cutout_size', type=int, default=None)  # 16
    parser.add_argument('-search_autoaugment', action='store_true', default=False)
    parser.add_argument('-search_num_work', type=int, default=12, help='the number of the data worker.')

    # ***************************  optimization setting******************
    parser.add_argument('-search_epochs', type=int, default=100)  # 35 # 30 # 50	#epochs for each solutions
    parser.add_argument('-search_lr_max', type=float, default=0.1)  # 0.025 NAO
    parser.add_argument('-search_lr_min', type=float, default=0.001)  # 0 for final training
    parser.add_argument('-search_momentum', type=float, default=0.9)
    parser.add_argument('-search_l2_reg', type=float, default=3e-4)  # 5e-4 for final training
    parser.add_argument('-search_grad_bound', type=float, default=5.0)
    parser.add_argument('-search_train_batch_size', type=int, default=512)  # 512  #raw 128
    parser.add_argument('-search_eval_batch_size', type=int,
                        default=200)  # 1000 #raw 500 ;IF USE MetaLR & Surrogate, 100?
    parser.add_argument('-search_steps', type=int, default=50000)
    # ***************************  structure setting******************
    parser.add_argument('-search_use_aux_head', action='store_true', default=True)
    parser.add_argument('-search_auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-search_layers', type=int, default=1)  # 3 for final Network
    parser.add_argument('-search_keep_prob', type=float, default=0.6)  # 0.6 also for final training
    parser.add_argument('-search_drop_path_keep_prob', type=float,
                        default=0.8)  # None 会在训练时提高 精度 和速度, 0.8等 更加耗时但最终训练会提升
    parser.add_argument('-search_channels', type=int, default=16)  # 24:48 for final training
    parser.add_argument('-search_channels_double', action='store_true',
                        default=False)  # False for Cifar, True for ImageNet model

    args = parser.parse_args()
    args.search_steps = int(np.ceil(45000 / args.search_train_batch_size)) * args.search_epochs
    args.save = '{}/EMO_search_{}'.format(args.save, time.strftime("%Y-%m-%d-%H-%M-%S"))

    create__dir(args.save)

    # ===================================  logging  ===================================
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/logs.log'.format(args.save),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

    logging.info("[Experiments Setting]\n" + "".join(
        ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))

    # ----------------------------------- logging  -------------------------------------

    # ===================================  random seed setting  ===================================
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    # -----------------------------------  random seed setting  -----------------------------------

    EMO_NAS = EMO(args, visualization=True)
    EMO_NAS.Main_loop()
