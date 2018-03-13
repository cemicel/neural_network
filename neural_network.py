import numpy as np
from matplotlib  import pyplot as plt

path = '/Users/volodymyrkepsha/Documents/Study/Python/Projects/Iris_NN_sig_grad_dic_β/train.txt'

#matrix_train = np.genfromtxt(path, delimiter=",", dtype=[('value', '4float64'), ('name', 'U16')])

matrix_train = np.genfromtxt(path_, delimiter=",", dtype=[('value', '4float64'), ('name', 'U16')]).random.shufle()




def show_error(errors ,iteration):
    pass


class NetWork(object):
    def __init__(self, matrix, input_weight, hidden_list_arg, targets, online=False, batch_size=None):
        """ initialization of neural network.

        """
        if online:
            batch_size = None

        self.batch_size = batch_size
        self.online_flag = online

        self.list_errors = []

        # architecture for nets
        self.number_of_units = [input_weight]
        if hidden_list_arg[0] != 0:
            self.number_of_units.extend(hidden_list_arg)
            self.number_of_units.append(len(targets))
            self.is_one_layer = False
        else:
            self.number_of_units.append(len(targets))
            self.is_one_layer = True

        self.targets = targets

        # 'error' - error of the network
        self.error = 0

        # input data
        self.input = [x[0] for x in matrix]

        # defining weight matrices
        self.weight_matrices = None
        # defining of result matrices
        self.results = None
        # defining error input
        self.error_out = np.zeros(len(targets))
        # defining of delta matrices
        self.deltas = None
        # bias
        self.biases = np.ones(len(hidden_list_arg) - 1)

        # hidden layer
        if hidden_list_arg[0] != 0:
            # initializing weight's matrices
            self.weight_matrices = [(np.random.rand(self.number_of_units[c], i)) for c, i in
                                    enumerate(self.number_of_units, start=1) if c < len(self.number_of_units)]
        else:
            # for one layer network
            self.weight_matrices = [np.random.rand(len(targets), input_weight)]
        # mini-batch

        if batch_size is not None:
            self.deltas = np.zeros((batch_size, len(targets)))
            self.results = [np.zeros((batch_size, i)) for i in self.number_of_units[1:]]
            self.error_out = np.zeros(batch_size, len(targets))
            # online
        elif online:
            self.deltas = np.zeros(len(targets))
            self.results = [np.zeros(i) for i in self.number_of_units[1:]]

            # full-batch
        else:
            self.deltas = np.zeros((len(matrix), len(targets)))
            self.results = [np.zeros((len(matrix), i)) for i in self.number_of_units[1:]]
            self.error_out = np.zeros(len(matrix) * len(targets)).reshape(len(matrix), len(targets))

    def feed_forward_batch(self, matrix):
        """         batch            """
        input_temp = self.input

        for l_c, layer in enumerate(self.weight_matrices):

            for input_c, input_v in enumerate(input_temp):
                re = self.activationSigmoidFunction(layer, input_v)
                self.results[l_c][input_c] = re
                # for output layer
                if l_c == len(self.weight_matrices) - 1:
                    self.deltas[input_c] = self.delta_output([1 if i == matrix[input_c][1] else 0 for i in targets_],
                                                             re)

                    # print([1 if i == matrix[input_c][1] else 0 for i in targets_] , re)
                    error = ([1 if i == matrix[input_c][1] else 0 for i in targets_] - re)

                    self.error_out[input_c] = error

                    self.error += np.sum(([1 if i == matrix[input_c][1] else 0 for i in targets_] - re) ** 2)

            input_temp = self.results[l_c]

    def feed_forward_online(self, matrix):
        """         online            """
        input_temp = self.input

        for input_c, input_v in enumerate(input_temp):

            for w_m_c, weight_matrix in enumerate(self.weight_matrices):

                re = self.activationSigmoidFunction(weight_matrix, input_v, 0.5)

                self.results[w_m_c] = re

                input_v = self.results[w_m_c]

                temp_error = np.zeros(3)
                # for output layer

                if w_m_c == len(self.weight_matrices) - 1:
                    temp_error = ([1 if i == matrix[input_c][1] else 0 for i in targets_] - re)

                    self.error_out = temp_error

                    # print(re,[1 if i == matrix[input_c][1] else 0 for i in targets_],self.error_out)


                    temp_error = [i ** 2 for i in temp_error]

                    self.error += np.sum(temp_error)

                    # self.deltas = self.delta([1 if i == matrix[input_c][1] else 0 for i in targets_], re)
                    self.deltas = self.delta(re)

                    self.gradient_step_online(input_c)

    def feed_forward_mini_batch(self, matrix, mini_batch):
        """         mini batch            """
        input_temp = mini_batch

        for l_c, layer in enumerate(self.weight_matrices):

            for input_c, input_v in enumerate(input_temp):
                re = self.activationSigmoidFunction(layer, input_v)

                self.results[l_c][input_c] = re

                # for output layer
                if l_c == len(self.weight_matrices) - 1:
                    self.deltas[input_c] = self.delta_output(
                        [1 if i == matrix[input_c][1] else 0 for i in targets_], re)

            input_temp = self.results[l_c]

    def run(self, matrix, iterations):

        if self.online_flag:
            for i in range(iterations):
                self.feed_forward_online(matrix)
                # print(network.error)
                self.error = 0
                # print(self.deltas)

        elif self.batch_size is None:

            for i in range(iterations):
                self.feed_forward_batch(matrix)
                self.gradient_step_batch()
                print(self.error)
                self.list_errors.append(self.error)
                self.error = 0



        elif self.batch_size is not None:
            batch = self.batch_size
            b_range = len(self.input) / batch
            for i in range(int(b_range)):
                self.feed_forward_mini_batch(matrix, self.input[i:i + batch])

    # under developing
    def gradient_step_online(self, input_counter):
        pass

    def test(self, element):
        return self.activationSigmoidFunction(self.weight_matrices[-1],
                                              self.activationSigmoidFunction(self.weight_matrices[0], element))

    def gradient_step_batch(self, learning_rate=0.175):

        ''' gradient step for ful batch learning. For this moment valid only for one hidden layer. '''

        # for one layer network
        if self.is_one_layer:
            dW = np.zeros((np.shape(self.weight_matrices)))
            for i in range(len(self.input)):
                p = np.outer(self.deltas[i], self.input[i])
                dW += p
            dW = dW * learning_rate
            print(dW)
            self.weight_matrices -= dW

        else:
            # dW - initialize delta weights for out put layer
            dW = np.zeros((np.shape(self.weight_matrices[-1])))
            # derivative of results of hidden layer
            der_hidden = self.delta(self.results[0])
            # dW - initialize delta weights for hidden put layer
            dWh = np.zeros((np.shape(self.weight_matrices[0])))

            for i in range(len(self.input)):
                dW += np.outer(self.deltas[i], self.results[-2][i])
                dWh += np.outer((self.deltas[i].dot(self.weight_matrices[-1])) * der_hidden[i], self.input[i])

            dW = dW * learning_rate

            self.weight_matrices[-1] -= dW * learning_rate
            self.weight_matrices[0] -= dWh * learning_rate


    def __repr__(self):
        if self.hidden is None:
            return 'neural net with 0 hidden layers'
        else:
            return 'neural net with {0} hidden layers'.format(len(self.hidden))


    @staticmethod
    def delta_output(target, output):
        return output * (1 - output) * (output - target)


    @staticmethod
    def delta(output):
        return output * (1 - output)


    @staticmethod
    def activationSigmoidFunction(layer_weights, input_vector, learning_rate=0.85):
        return 1 / (1 + np.exp(-layer_weights.dot(input_vector) * learning_rate))


    @staticmethod
    def sigmoid_derivative(result_of_act_func):
        return result_of_act_func * (1 - result_of_act_func)  # weight hidden targets  # if __name__ is '__main__':


targets_ = set(i[1] for i in matrix_train)
targets_ = list(targets_)


network = NetWork(matrix_train, 4, [5], targets_)
print(network.number_of_units)

network.run(matrix_train, 200)
print(network.list_errors)

print(network.test(np.array([4.8, 3.0, 1.4, 0.3])))

plt.plot(range(200),network.list_errors)
plt.show()