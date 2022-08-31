import math

import functions
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

import sympy as sp

if __name__ == '__main__':
    solver = functions.solver()
    # # week 1
    # # 1_7
    predicted_classes = [1,0,1,1,0,1,0]
    true_classes = [1,1,0,1,0,1,1]
    # solver.draw_confusion_matrix(predicted_classes,true_classes)

    # week 2
    # 2_1
    w = [2, 1]
    w0 = -5
    feature_vector = [[1, 1],
                      [2, 2],
                      [3, 3],
                      [0,0]]

    # solver.dichotimizer(w, w0, feature_vector)

    # 2_2
    # augmentatation format [w0,w1.w2], feature = [1,x1,x2]
    a = [-5, 2, 1]
    y = [[1, 1, 1], [1, 2, 2], [1, 3, 3]]
    # solver.augmented_dichotimiezer(a, y)





    # 2_5
    a = [-3, 1, 2, 2, 2, 4]
    y =[
        [1,0,-1,0,0,1],
        [1,1,1,1,1,1]
    ]
    # solver.augmented_dichotimiezer(a, y)

    # 2_6
    x = [[1, 1, 5],
         [1, 2, 5],
         [-1, -4, -1],
         [-1, -5, -1]]
    label = [1, 1, -1, -1]
    a_t = [-25, 6, 3]
    learning_rate = 1
    x_not = [[1, 1, 5],
             [1, 2, 5],
             [1, 4, 1],
             [1, 5, 1]]
    # print(solver.batch_perceptron_learning_algorithm_with_normalisation(x_list=x, a_t=a_t, learning_rate=learning_rate))

    # 2_7
    x = [[1, 1, 5],
         [1, 2, 5],
         [-1, -4, -1],
         [-1, -5, -1]]
    label = [1, 1, -1, -1]
    a_t = [-25, 6, 3]
    learning_rate = 1
    x_not = [[1, 1, 5],
             [1, 2, 5],
             [1, 4, 1],
             [1, 5, 1]]
    # solver.sequential_perceptron_learning_algorithm_with_sample_normalization(x_list=x, a_t=a_t, learning_rate=learning_rate)

    # 2_9
    # x_not without sample normaliztion 在x=-1的时候和x=1是整个全部都是相反数
    x = [[1, 0, 2],
         [1, 1, 2],
         [1, 2, 1],
         [-1, 3, -1],
         [-1,2,1],
         [-1,3,2]]
    label = [1, 1, 1, -1, -1, -1]
    a_t = [1, 0, 0]
    learning_rate = 1
    x_not = [[1, 0, 2],
             [1, 1, 2],
             [1, 2, 1],
             [1, -3, 1],
             [1,-2,-1],
             [1,-3,-2]]
    # solver.sequential_perceptron_learning_algorithm_without_sample_normalization(x_list=x_not, a_t=a_t, learning_rate=learning_rate,label=label)



    # 2_10
    x = [[1, 0, 2],
         [1, 1, 2],
         [1, 2, 1],
         [-1, 3, -1],
         [-1, 2, 1],
         [-1, 3, 2]]
    label = [1, 1, 1, -1, -1, -1]
    a_t = [1, 0, 0]
    learning_rate = 1
    x_not = [[1, 0, 2],
             [1, 1, 2],
             [1, 2, 1],
             [1, -3, 1],
             [1, -2, -1],
             [1, -3, -2]]
    # solver.sequential_perceptron_learning_algorithm_with_sample_normalization(a_t=a_t, x_list=x, learning_rate=learning_rate)

    # 2_11
    x = [[1, 1, 1],
         [1, 2, 0],
         [1, 0, 2],
         [1, -1, 1],
         [1, -1, -1]]
    label = [1, 1, 2, 2, 3]
    a_t = [0, 0, 0]
    class_numbers = len(set(label))
    learning_rate = 1
    # solver.Sequential_Multiclass_Perceptron_Learning_algorithm_without_sample_normalization(a_t= a_t, x_list=x, label=label, learning_rate=learning_rate,class_numbers=class_numbers)


    ####
    x = [[1, 0, 1],
         [1, 1, 0],
         [1, 0.5, 1.5],
         [1, 1, 1],
         [1, -0.5, 0]]
    label = [1, 1, 2, 2, 3]
    a_t = [0, 0, 0]
    class_numbers = len(set(label))
    learning_rate = 1
    # solver.Sequential_Multiclass_Perceptron_Learning_algorithm_without_sample_normalization(a_t= a_t, x_list=x, label=label, learning_rate=learning_rate,class_numbers=class_numbers)



    # 2_12
    y = [[1,0,2],
         [1,1,2],
         [1,2,1],
         [-1,3,-1],
         [-1,2,1],
         [-1,3,2]]

    b = [1,1,1,1,1,1]
    # solver.pseudoinverse_with_sample_normalization(y,b)

    # 2_13
    y = [[1, 0, 2],
         [1, 1, 2],
         [1, 2, 1],
         [-1, 3, -1],
         [-1, 2, 1],
         [-1, 3, 2]]

    # b = [2, 2, 2, 1, 1, 1]
    b = [1,1,1,2,2,2]
    # solver.pseudoinverse_with_sample_normalization(y, b)

    # 2_14
    x_t = [[1, 0, 2],
           [1, 1, 2],
           [1, 2, 1],
           [-1, 3, -1],
           [-1, 2, 1],
           [-1, 3, 2]]
    b_t = [1,1,1,1,1,1]
    learning_rate = 0.1
    a_t = [[1, 0, 0]]
    # solver.sequential_widrow_hoff_learning_with_sample_normalization(a_t, b_t, x_t, learning_rate, 2)

    # 2_15
    training_x = [[0.3, 0.35],
         [0.3, 0.28],
         [0.24, 0.2],
         [0.2,0.32],
         [0.12,0.25]]
    true_label = [1,2,2,3,3]

    new_feature = [
        [0.2,0.25]]

    # solver.knn(training_x, true_label,new_feature,k=3)

    # week 3
    # 3_2
    # 注意这里要使用augmented的形式，w =[-theta,w]; x=[1,x]
    # 函数是w1x1+w2x2 = theta
    w = [0,0.1,-0.5,0.4]
    x =[[1,0.1,-0.5,0.4],
        [1,0.1,0.5,0.4]]
    threshold = 0
    alpha = 0.1
    # solver.calculate_output_of_a_neuron_with_augmentation_without_sample_normalization(w=w,x=x,activation_function='Heaviside',threshold = threshold,alpha=0.1)


    # 3.3
    w = [[-1.5, 2]]
    x = [[1, 0],
         [1, 1]]
    y = [1, 0]
    # solver.sequential_delta_learning_algorithm_stop_when_all_match(w, y, x)

    w = [[-1.5, 2]]
    x = [[1, 0],
         [1, 1]]
    y = [1, 0]
    # solver.sequential_delta_learning_algorithm_stop_when_all_match(w, y, x)


    # 3.4
    # w传传入参数一定要注意，theta是-theta，别错了
    # w = [-theta, w1,w2]
    # x = [1,x1,x2]
    # y可以是0
    w = [[1,3, 0.5]]
    x = [[1, 2,-1],
         [1,-1, 0],
         [1,0,0],
         [1,1,1],
         [1,0,-1]]
    y = [0, 1,1,0,1]
    # solver.sequential_delta_learning_algorithm_stop_when_all_match(w, y, x,learning_rate=1)



    # 3.5
    # w传传入参数一定要注意，theta是-theta，别错了
    # w = [-theta, w1,w2]
    # x = [1,x1,x2]
    w = [[0.5, 1, 1]]
    x = [[1, 0, 0],
         [1, 0, 1],
         [1, 1, 0],
         [1, 1, 1]]
    y = [0, 0, 0, 1]
    # solver.sequential_delta_learning_algorithm_stop_when_all_match(w, y, x)

    # 3_6:
    x = [[1,0,2],
         [1,1,2],
         [1,2,1],
         [1,-3,1],
         [1,-2,-1],
         [1,-3,-2]]
    w = [[1,0,0]]
    label = [1,1,1,0,0,0]
    learning_rate = 1
    # solver.sequential_delta_learning_algorithm_stop_when_all_match(w=w,label=label,x=x,learning_rate=learning_rate)

    # ppt
    x = [[1,0,0],
         [1,1,0],
         [1,2,1],
         [1,0,1],
         [1,1,2]]
    label = [1,1,1,0,0]
    learning_rate = 1
    w = [-1.5,5,-1]
    # solver.sequential_delta_learning_algorithm_stop_when_all_match(w=w,x=x,label=label,learning_rate=learning_rate)

    # 3_7
    w = [[1, 1, 0],
         [1, 1, 1]]
    x_t = [[1, 1, 0]]
    y_t = [[0, 0]]
    alpha = 0.25
    # solver.negative_feedback_neural_network_update_e_only(weights=w, update_rate=alpha, x_t=x_t, y_t=y_t, iteration=5)

    # 3_8
    w=[[1,1,0],
       [1,1,1]]
    x_t = [[1,1,0]]
    alpha = 0.5
    y_t = [[0,0]]
    # solver.negative_feedback_neural_network_update_e_only(weights=w,update_rate=alpha,x_t = x_t,y_t=y_t,iteration=5)

    # 3_9
    # sig_1 动y
    # sig_2 动 e
    sig_1 = .01
    sig_2 = .01
    x_t = [1,1,0]
    w=[[1,1,0],
       [1,1,1]]
    y_t = [0,0]
    iteration = 5
    # solver.Regulatory_feedback(w, x_t, y_t, sig_1, sig_2, iteration=iteration)

    # week 4
    # 4_1 4_2
    architecture = [2,4,5,3]
    # solver.fully_connected_neuron_network_architecture(architecture)

    # 4_4
    # Symmetric hard limit
    def symmetric_hard_limit(z):
        if z >= 10e-5:
            return 1
        else:
            return -1
    # linear transfer
    def linear_transfer(z):
        return z
    # Symmetric sigmoid transfer function
    def symmetric_sigmoid(z):
        return 2.0 / (1 + np.exp(-2 * z)) - 1
    # Logarithmic sigmoid transfer function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Radial basis transfer function
    def radial_basis(z):
        return np.exp(-1 * np.square(z))



    def d_symmetric_sigmoid(z):
        return 4.0*np.exp(-2*z)/np.square(1+np.exp(-2*z))

    def d_output(true,predicted):
        return predicted - true




    input = [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [1, 1, 0, 0]]
    W_hidden_input =[[-0.7057 ,1.9061 ,2.6605, -1.1359],
                     [0.4900, 1.9324, -0.4269, -5.1570],
                     [0.9438 ,-5.4160, -0.3431 ,-0.2931]]
    bias_hidden_input = [4.8432,0.3973,2.1761]
    W_output_hidden = [[-1.1444, 0.3115 ,-9.9812],
                       [0.0106, 11.5477, 2.6479]]
    bias_bias_output_hidden = [2.5230,2.6463]
    a_input = linear_transfer
    a_hidden = symmetric_sigmoid
    a_output = sigmoid

    # solver.feed_forward_NN(input_array_t=input,W_h_i=W_hidden_input, W_o_h=W_output_hidden,b_h_i_t=bias_hidden_input,b_o_h_t=bias_bias_output_hidden,a_input=linear_transfer,a_hidden=a_hidden,a_output=a_output,verbose=True)


    # q5
    input = [[0.1,0.9]]
    W_hidden_input = [[0.5,  0],
                      [0.3, -0.7 ]]

    bias_hidden_input = [0.2, 0]
    W_output_hidden = [[0.8, 1.6]]
    bias_bias_output_hidden = [-0.4]
    a_input = linear_transfer
    a_hidden = symmetric_sigmoid
    a_output = symmetric_sigmoid

    # solver.feed_forward_NN(input_array_t=input, W_h_i=W_hidden_input, W_o_h=W_output_hidden, b_h_i_t=bias_hidden_input,
    #                        b_o_h_t=bias_bias_output_hidden, a_input=linear_transfer, a_hidden=a_hidden,
    #                        a_output=a_output, verbose=True)




    # print("4_5")
    # print(solver.q4_backpropagation(0.1, 0.9))

    def gaussian_hidden_function(distance, sigma):
        """

        :param distance:
        :param sigma: sigma here has to be greater than 0
        :return:
        """
        return np.exp(-(distance ** 2) / (2 * (sigma ** 2)))

    def Multi_quadric_function(distance,sigma):
        return np.sqrt(np.square(distance) + np.square(sigma))

    def Generalized_multi_quadric_function(distance,sigma,beita = 0.5):
        return np.power(np.square(distance) + np.square(sigma),beita)

    def Inverse_multi_quadric_function(distance,sigma):
        return np.power(np.square(distance)+np.square(sigma),-0.5)

    def Generalized_inverse_multi_quadric_function(distance,sigma,beita):
        return np.power(np.square(distance) + np.square(sigma),-1*beita)

    def Thin_plate_spline_function(distance,sigma):
        return np.square(distance)*np.log(distance)

    # 4_6
    input_layer = [[0.90,0.60],
                    [0.30,0.70],
                    [0.40,1.00],
                    [0.30,0.00],
                   [0.80,0.80],
                   [0.80,0.20],
                   [0.40,0.50],
                   [0.90,0.00],
                   [0.10,0.40],
                   [0.10,0.60]]
    centers = [[0.30,0.00],
                [0.90,0.60],
               [0.10,0.60]]
    way = "max" # 也可使是avg
    label_t = [[1, 0, 0, 0,0,0,0,1,1,1]]  # 2-D
    weights, sigma = solver.radial_basis_function(input_layer, centers, way,
                                                  gaussian_hidden_function,
                                                  label_t)





    input_layer_lgt_4_6_d = [[0.5, -0.1],
                             [-0.2, 1.2],
                             [0.8, 0.3],
                             [1.8, 1.6]]
    # solver.radial_basis_function_given_weights(input_layer_lgt_4_6_d, weights, centers, sigma,
    #                                            gaussian_hidden_function)



    # 4_7
    input_ = [[0.0500],
              [0.2000],
              [0.2500],
              [0.3000],
              [0.4000],
              [0.4300],
              [0.4800],
              [0.6000],
              [0.7000],
              [0.8000],
              [0.9000],
              [0.9500]]
    # centers_ = [[0.1667],
    #             [0.35],
    #             [0.5525],
    #             [0.8833]]
    centers_ = [[0.2],
                [0.6],
                [0.9]]
    way = "avg"
    label_t = [[0.0863, 0.2662, 0.2362, 0.1687,0.1260,0.1756,0.3290,0.6694,0.4573,0.3320,0.4063,0.3535]]
    # weights, sigma = solver.radial_basis_function(input_, centers_, way,
    #                                               gaussian_hidden_function,
    #                                               label_t)

    # week 5
    # 5_4
    net_j = [[1, 0.5, 0.2],
             [-1, -0.5, -0.2],
             [0.1, -0.1, 0]]
    # Relu
    # print(solver.relu(net_j))
    # LRelu
    # print(solver.lrelu(net_j, 0.1))
    # tanh
    # print(solver.tanh(net_j))
    # d) Heaviside function where each neuron has a threshold of 0.1:
    # print(solver.heaviside_matrix(net_j, 0.1))

    X = [
        [[0.9, 1.0, -1.0],
         [0.6, 0.5, 0.6],
         [0.7, -0.9, 0.4]],

        [[0.9, 0.6, -0.8],
         [0.7, 0.8, -0.1],
         [0.2, -0.2, -0.6]],

        [[-0.8, -0.3, -1.0],
         [-0.9, -0.7, 0.0],
         [0.0, 0.6, -0.5]]]

    # solver.cnn_batch_normalisation(X, beta=0.1, gamma=0.4, epsilon=0.2)

    X_lgt_5_6 = [[[0.2, 1, 0],
                  [-1, 0, -0.1],
                  [0.1, 0, 0.1]],

                 [[1, 0.5, 0.2],
                  [-1, -0.5, -0.2],
                  [0.1, -0.1, 0]]]

    H_lgt_5_6 = [[[1, -.1],
                  [1, -.1]],

                 [[0.5, 0.5],
                  [-0.5, -0.5]]]


    test_H = H_lgt_5_6 = [[[1, 0,-.1],
                           [0, 0, 0],
                           [1, 0,-.1]]]

    test_X = [[[0.2, 1, 0],
                  [-1, 0, -0.1],
                  [0.1, 0, 0.1]]
              ]

    # solver.cross_correlation_with_stride_dilation(test_X,test_H,padding=0)
    # 注意，只有padding能用，stride 1，dilation 1; padding 任意值，stride 1，dilation 1直接套公式
    #  stride大于 1 ，划X,
    # dilation大于1，动 H
    # solver.cross_correlation_with_stride_dilation(X_lgt_5_6, H_lgt_5_6, padding=0)
    # solver.cross_correlation_with_stride_dilation(X_lgt_5_6, H_lgt_5_6, padding=1)

    X_lgt_5_7 = [[[0.2, 1, 0],
                  [-1, 0, -0.1],
                  [0.1, 0, 0.1]],

                 [[1, 0.5, 0.2],
                  [-1, -0.5, -0.2],
                  [0.1, -0.1, 0]],

                 [[0.5, -.5, -0.1],
                  [0, -0.4, 0],
                  [0.5, 0.5, 0.2]]]
    masks_lgt_5_7 = [1, -1, 0.5]
    # solver.convolution1X1(X_lgt_5_7, masks_lgt_5_7)


    # week 6
    # 6_4
    # LGT 6 GAN
    real_x_lgt_6_4 = [[1, 2],
                      [3, 4]]

    fake_x_lgt_6_4 = [[5, 6],
                      [7, 8]]

    theta1_lgt_6_4 = 0.1
    theta2_lgt_6_4 = 0.2

    discriminator_lgt_6_4 = lambda x: 1 / (1 + math.exp(-(theta1_lgt_6_4 * x[0] - theta2_lgt_6_4 * x[1] - 2)))
    # solver.gan_generator_and_discriminator(fake_x_lgt_6_4, real_x_lgt_6_4, discriminator_lgt_6_4)


    # Week 7
    # 7_4
    # 7_5 proportion
    #K_L Transformation
    # feature 按照行传进来
    X_features = [[5.0,5.0,4.4,3.2],
                  [6.2,7.0,6.3,5.7],
                  [5.5,5.0,5.2,3.2],
                  [3.1,6.3,4.0,2.5],
                  [6.2,5.6,2.3,6.1]]
    # print(solver.kl_transform(X_features,2))

    # X_features = [[0, 1],
    #               [3, 5],
    #               [5, 4],
    #               [5 ,6],
    #               [8,7],
    #               [9,7]]
    # print(solver.kl_transform(X_features, 1))

    # 7_7 Oja's rule
    # 多跑一个epoch就能算apply后的结果了
    X_features = [[0, 1],
                  [3, 5],
                  [5, 4],
                  [5 ,6],
                  [8,7],
                  [9,7]]
    w = [-1,0]
    # print(solver.oja_rules_sequential(w,X_features,0.01,1))
    # print(solver.oja_rules(w, X_features, 0.01,2))



    # LDA
    class_vector  = [
                        [[1,2],
                        [2,1],
                        [3,3]],

                        [[6,5],
                        [7,8]],
                     ]
    w=[-1,5]
    w2 = [2,-3]
    # print(solver.fisher_method(class_vector,w2))

    # 7_11 extreme_learning_machine
    random_matrix = [[-0.62, 0.44, -0.91],
                     [-0.81, -0.09, 0.02],
                    [0.74, -0.91, -0.60],
                     [-0.82, -0.92, 0.71],
                    [-0.26, 0.68, 0.15],
                     [0.80, -0.94, -0.83]]
    x_features = [[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]]
    output_neuron = [0, 0, 0, -1, 0, 0, 2]
    # solver.extreme_learning_machine(random_matrix, x_features, output_neuron)

    # 7_12
    y_t = [[1, 0, 0, 0, 1, 0, 0, 0]]
    V_t = [[0.4, 0.55, 0.5, -0.1, -0.5, 0.9, 0.5, 0.45],
           [-0.6, -0.45, -0.5, 0.9, -0.5, 0.1, 0.5, 0.55]]
    x = [-0.05, -0.95]
    # solver.sparse_coding(x, y_t, V_t)
    y2_t = [[0, 0, 1, 0, 0, 0, -1, 0]]
    # solver.sparse_coding(x, y2_t, V_t)
    # y3_t = [[1, 0, 0, 0, 1, 0, 0, 0]]
    # solver.sparse_coding(x, y3_t, V_t)
    # y4_t = [[0, 0, 0, -1, 0, 0, 0, 0]]
    # solver.sparse_coding(x, y4_t, V_t)

    y_t = [[0, 0.5, 1, 0]]
    V_t = [[1,1,2,1],
           [-4,3,2,-1]]
    x = [2,3]
    # solver.sparse_coding(x, y_t, V_t)


    # week 8
    # X = [[3, 3], [9, 7], [5, 5], [7, 9],
    #      [-1, -2], [-1, -4], [-3, -4], [-3, -2]]
    # Y = [1, 1, 1, 1, -1, -1, -1, -1]

    X = [[3,1],[3,-1],[7,1],[8,0],
         [1,0],[0,1],[-1,0],[-2,0]]
    Y = [1,1,1,1,-1,-1,-1,-1]
    # clf = solver.svm(X,Y)
    X = [[-3, 9], [-2, 4], [6, 36], [8, 64],
         [-1, 1], [0, 0], [1, 1], [2, 4]]
    Y = [1, 1, 1, 1, -1, -1, -1, -1]
    # clf = solver.svm_([[-2,4],[-1,1],[2,4]],[1,-1,-1])






    x_lgt_8_3_svm = [[2, 2],
                     [2, -2],
                     [-2, -2],
                     [-2, 2],
                     [1, 1],
                     [1, -1],
                     [-1, -1],
                     [-1, 1]]
    class_x_lgt_8_3 = [1, 1, 1, 1, -1, -1, -1, -1]


    def mapping_function(datum):
        if np.linalg.norm(datum) >= 2:
            new_datum = [4 - datum[1] / 2 + abs(datum[0] - datum[1]), 4 - datum[0] / 2 + abs(datum[0] - datum[1])]
        else:
            new_datum = [datum[0] - 2, datum[1] - 3]
        return new_datum


    new_data = solver.mapping_to_another_space(x_lgt_8_3_svm, mapping_function)
    # print(new_data)

    # week 9
    x_lgt9_1 = [[1, 0],
                [-1, 0],
                [0, 1],
                [0, -1]]
    label_lgt9_1 = [1, 1, -1, -1]
    weak_functions = [lambda x: 1 if x[0] > -0.5 else -1,
                      lambda x: -1 if x[0] > -0.5 else 1,
                      lambda x: 1 if x[0] > 0.5 else -1,
                      lambda x: -1 if x[0] > 0.5 else 1,
                      lambda x: 1 if x[1] > -0.5 else -1,
                      lambda x: -1 if x[1] > -0.5 else 1,
                      lambda x: 1 if x[1] > 0.5 else -1,
                      lambda x: -1 if x[1] > 0.5 else 1]
    # solver.adaboost_algorithm(x_lgt9_1, label_lgt9_1, 3, weak_functions)
    sgn_lgt10_2 = lambda x: 1 if x >= 0 else -1
    maximum_training_error = 0.5
    # solver.bagging(x_lgt9_1, label_lgt9_1, weak_functions, maximum_training_error, sgn_lgt10_2)


    # week 10
    feature_vector_array = [[-1,3],[1,2],[0,1],[4,0],[5,4],[3,2]]
    # solver.naive_agglomerative_hierarchical_clustering(feature_vector_array, k=3)

    def euclidean_distance(array1, array2):
        array1 = np.array(array1)
        array2 = np.array(array2)
        return np.sqrt(np.sum(np.square(np.absolute(array1 - array2))))

    def single_linkage(cluster_1, cluster_2):
        minimum_dis = float("inf")
        for node_1 in cluster_1:
            for node_2 in cluster_2:
                if minimum_dis > euclidean_distance(node_1, node_2):
                    minimum_dis = euclidean_distance(node_1, node_2)
        return minimum_dis


    def centroid_distance(cluster1, cluster2):
        cluster1_center = np.mean(cluster1, axis=0)
        cluster2_center = np.mean(cluster2, axis=0)
        return euclidean_distance(cluster1_center, cluster2_center)

    feature_vector_array = [[[-1, 3]],
                            [[1, 2]],
                            [[0, 1]],
                            [[4, 0]],
                            [[5, 4]],
                            [[3, 2]]]
    # x = solver.agglomerative_hierarchical_clustering(feature_vector_array, 3, distance_function=single_linkage)
    # print(x)
    # print("centroid_distance")

    feature_vector_array = [[[-1, 3]], [[1, 2]], [[0, 1]], [[4, 0]], [[5, 4]], [[3, 2]]]
    # x = solver.agglomerative_hierarchical_clustering(feature_vector_array, 3, distance_function=centroid_distance)
    # print(x)

    # k menas
    data_set = [[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]]
    initial_center = [[-1, 3], [5,1],[3,0]]
    # data, centers = solver.kmeans(data_set, initial_center)


    ##
    # k menas
    data_set = [[5, 1], [1, 1], [4, 7], [6, 4], [4, 6],[5,8]]
    initial_center = [[6, 4], [1,1]]
    # data, centers = solver.kmeans(data_set, initial_center)


    feature_vector_array = [[[1, 0]], [[0, 2]], [[1, 3]], [[3, 0]], [[3, 1]]]

    # x = solver.agglomerative_hierarchical_clustering(feature_vector_array, 2, distance_function=single_linkage)
    # print(x)

    #K_L Transformation
    # feature 按照行传进来
    X_features = [[4,2,2],
                  [0,-2,2],
                  [2,4,2],
                  [-2,0,2]]
    # print(solver.kl_transform(X_features,2))

    x_features = [[5.0, 5.0, 4.4, 3.2],
                  [6.2, 7.0, 6.3, 5.7],
                  [5.5, 5.0, 5.2, 3.2],
                  [3.1, 6.3, 4.0, 2.5],
                  [6.2, 5.6, 2.3, 6.1]]
    eigen_values = [0.0, 0.71, 1.90, 3.21]
    eigen_vectors_t = [[-0.59, -0.56, 0.25, 0.52],
                       [0.55, -0.78, 0.12, -0.27],
                       [0.11, 0.25, 0.96, -0.07],
                       [0.58, 0.12, -0.04, 0.81]]
    # solver.kl_transform_given_values(x_features, eigen_values, eigen_vectors_t, 2)


    # competitive learning
    def l2_norm(vector_1, vector_2):
        assert len(vector_1) == len(vector_2)
        # TO make this numpy array
        vector_1 = np.array(vector_1)
        vector_2 = np.array(vector_2)
        l2 = np.power(np.sum(np.power((vector_1 - vector_2), 2)), 0.5)
        return l2

    initial_center = [[-0.5,1.5],[0,2.5],[1.5,0]]
    learning_rate = 0.1
    iteration = 5
    sample_with_order = [[0,5],[-1,3],[-1,3],[3,0],[5,1]]
    # ### full_sample =[[-1,3],[1,4],[0,5],[4,1],[3,0],[5,1],[0,-2]]
    # cluster_center = solver.competitive_learning_without_normalization(initial_center, learning_rate, iteration, sample_with_order,
    #                                            l2_norm)

    test = [[0,-2]]
    # print(solver.given_cluster_center_compute_its_belongs(cluster_center,test,l2_norm))

    # 注意在 with normalization的版本里，需要augmentation， x = [1,x]
    initial_center = [[1,-0.5, 1.5], [1,0, 2.5], [1,1.5, 0]]
    learning_rate = 0.1
    iteration = 5
    sample_with_order = [[1,0, 5], [1,-1, 3], [1,-1, 3], [1,3, 0], [1,5, 1]]
    # cluster_center = solver.competitive_learning_with_normalization(initial_center, learning_rate, iteration, sample_with_order,
    #                                            l2_norm)

    # 10_4
    x = [[-1, 3],
         [1, 4],
         [0, 5],
         [4, -1],
         [3, 0],
         [5, 1]]
    theta = 3
    n = 0.5
    # pickup order 的初始化要全部-1 [3 1 1 5 6] -> [2,0,0,4,5]
    pickup_order = [2, 0, 0, 4, 5]
    # solver.basic_leader_follower_clustering_without_normalisation(x, theta, n, pickup_order)

    # 10_5
    # 2个center，6个数据 data is s, m
    s = [[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]]
    miu = [[1, 0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0, 1]]
    k = 2
    b = 2
    # solver.fuzzy_kmeans(s, miu, k, b)



