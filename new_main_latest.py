import functions_latests
import numpy as np
import math


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


# Activation functions
def linear_function(array: list):
    return array


def relu(net):
    net = np.array(net)
    net[net < 0] = 0
    return net


def lrelu(net, alpha):
    net = np.array(net)
    net[net < 0] = net[net < 0] * alpha
    return net


def tansigmoid(net):
    net = (2 / (1 + np.exp(-2 * net))) - 1
    return net


def sigmoid(net):
    return 1 / (1 + np.exp(-net))


def sigmoid_tanh(net):
    return np.tanh(net)


def gaussian_hidden_function(distance, sigma):
    """

    :param distance:
    :param sigma: sigma here has to be greater than 0
    :return:
    """
    return np.exp(-(distance ** 2) / (2 * (sigma ** 2)))


def linear_hidden_function(distance, sigma):
    return distance


# Classification function
def symmetric_hard_limit(value):
    return 1 if value >= 1 else -1


if __name__ == '__main__':

    solver = functions_latests.solver()
    # batch_perceptron_learning_algorithm_with_augmentation_normalisation
    # Manually normalised and augmented
    x_lgt2_5 = [[1, 1, 5], [1, 2, 5], [-1, -4, -1], [-1, -5, -1]]
    # Since it's been normalised, actually no label is need.
    label_lgt2_5 = [1, 1, -1, -1]
    initial_a_t_lgt2_5 = [-25, 6, 3]
    solver.batch_perceptron_learning_algorithm_with_augmentation_normalisation(x_lgt2_5, initial_a_t_lgt2_5, 1)
    # Repeat the previous question, LGT 2_7
    solver.sequential_delta_learning_algorithm_stop_with_augmentation_normalisation(x_lgt2_5, initial_a_t_lgt2_5, 1)

    # LGT2_9
    x_lgt2_9 = [[1, 0, 2], [1, 1, 2], [1, 2, 1], [1, -3, 1], [1, -2, -1], [1, -3, -2]]
    label_lgt_2_9 = [1, 1, 1, -1, -1, -1]
    initial_a_t_lgt_2_9 = [1, 0, 0]
    solver.sequential_perceptron_learning_algorithm_without_normalisation(x_lgt2_9, initial_a_t_lgt_2_9, label_lgt_2_9)

    # Test
    x_lgt2_5_without_normalisation = [[1, 1, 5], [1, 2, 5], [1, 4, 1], [1, 5, 1]]
    solver.sequential_perceptron_learning_algorithm_without_normalisation(x_lgt2_5_without_normalisation,
                                                                          initial_a_t_lgt2_5, label_lgt2_5)
    # LGT 2_12
    y_t_lgt2_12 = [[1, 0, 2], [1, 1, 2], [1, 2, 1], [-1, 3, -1], [-1, 2, 1], [-1, 3, 2]]
    margin_t_lgt2_12 = [[1, 1, 1, 1, 1, 1]]
    solver.pseudo_inverse_with_margin(y_t_lgt2_12, margin_t_lgt2_12)

    # LGT 2_13, change the margin_t
    margin_t_lgt2_13_a = [[2, 2, 2, 1, 1, 1]]
    margin_t_lgt2_13_b = [[1, 1, 1, 2, 2, 2]]
    solver.pseudo_inverse_with_margin(y_t_lgt2_12, margin_t_lgt2_13_a)
    solver.pseudo_inverse_with_margin(y_t_lgt2_12, margin_t_lgt2_13_b)

    # LGT 2_14, the Widrow-Hoff Learning
    iteration_2_14 = 12
    initial_a_t_lgt2_14 = [[1, 0, 0]]
    solver.sequential_widrow_hoff_learning(initial_a_t_lgt2_14, margin_t_lgt2_12, y_t_lgt2_12, 0.1, 12)
    x_lgt2_15 = [{"class": 1, "value": [0.15, 0.35]},
                 {"class": 2, "value": [0.15, 0.28]},
                 {"class": 2, "value": [0.12, 0.20]},
                 {"class": 3, "value": [0.10, 0.32]},
                 {"class": 3, "value": [0.06, 0.25]}]
    new_item_lgt2_15 = [0.1, 0.25]
    solver.knn(x_lgt2_15, new_item_lgt2_15, 3)

    # LGT 3
    weights_lgt3_2 = [0.1, -0.5, 0.4]
    x1_lgt3_2 = [0.1, -0.5, 0.4]
    x2_lgt3_2 = [0.1, 0.5, 0.4]
    solver.nn_given_wights_and_input(x1_lgt3_2, weights_lgt3_2)
    solver.nn_given_wights_and_input(x2_lgt3_2, weights_lgt3_2)

    # LGT 3_3 Linear Threshold Unit
    augmented_x_1_lgt3_3 = [[1, 0], [1, 1]]
    theta_lgt3_3 = 1.5
    augmented_initial_w_lgt3_3 = [-theta_lgt3_3, 2]
    label_lgt3_3 = [1, 0]
    solver.sequential_delta_learning_rule(augmented_x_1_lgt3_3, augmented_initial_w_lgt3_3, label_lgt3_3, 1)

    # LGT 3_4 Linear Threshold Unit learning using batch delta learning rule.

    solver.batch_delta_learning_rule(augmented_x_1_lgt3_3, augmented_initial_w_lgt3_3, label_lgt3_3)

    # LGT 3_5 Linear with 2 inputs
    augmented_x_lgt_3_5 = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    theta_lgt3_5 = -.5
    augmented_initial_w_lgt3_5 = [-theta_lgt3_5, 1, 1]
    label_lgt3_5 = [0, 0, 0, 1]
    solver.sequential_delta_learning_rule(augmented_x_lgt_3_5, augmented_initial_w_lgt3_5, label_lgt3_5, 1)

    # LGT 3_6 Linearly separable data set
    # No need to do the normalisation
    augmented_x_lgt3_6 = [[1, 0, 2], [1, 1, 2], [1, 2, 1], [1, -3, 1], [1, -2, -1], [1, -3, -2]]
    theta_lgt3_6 = -1
    augmented_initial_w_lgt3_6 = [-theta_lgt3_6, 0, 0]
    learning_rate_lgt3_6 = 1
    label_lgt3_6 = [1, 1, 1, 0, 0, 0]
    solver.sequential_delta_learning_rule(augmented_x_lgt3_6, augmented_initial_w_lgt3_6, label_lgt3_6,
                                          learning_rate_lgt3_6)

    # Negative feedback 3_8
    weights_lgt3_8 = [[1, 1, 0], [1, 1, 1]]
    x_t_lgt3_8 = [1, 1, 0]
    update_rate_lgt3_8 = 0.25
    y_t_lgt3_8 = [0, 0]
    solver.negative_feedback_neural_network_update_e_only(weights_lgt3_8, update_rate_lgt3_8, x_t_lgt3_8, y_t_lgt3_8, 5)
    update_rate_lgt3_9 = 0.5
    solver.negative_feedback_neural_network_update_e_only(weights_lgt3_8, update_rate_lgt3_9, x_t_lgt3_8, y_t_lgt3_8, 5)

    # NN_given_weights LGT 4_4
    input_layer_t_lgt_4_4 = [[1, 1, 0, 0]]
    weights_lgt_4_4 = [[[-0.7057, 1.9061, 2.6605, -1.1359],
                        [0.4900, 1.9324, -0.4269, -5.1570],
                        [0.9438, -5.4160, -0.3431, -0.2931]],
                       [[-1.1444, 0.3115, -9.9812],
                        [0.0106, 11.5477, 2.6479]]
                       ]
    bias_lgt_t_4_4 = [[[4.8432, 0.3973, 2.1761]], [[2.5230, 2.6463]]]
    solver.NN_given_weights(input_layer_t_lgt_4_4, weights_lgt_4_4, bias_lgt_t_4_4,
                            [linear_function, tansigmoid, sigmoid])

    input_layer_lgt_4_6 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    centers_lgt_4_6 = [[0, 0], [1, 1]]
    way_lgt_4_6 = "max"
    label_t_lgt_4_6 = [[0, 1, 1, 0]]  # 2-D
    weights, sigma = solver.radial_basis_function(input_layer_lgt_4_6, centers_lgt_4_6, way_lgt_4_6,
                                                  gaussian_hidden_function,
                                                  label_t_lgt_4_6)

    input_layer_lgt_4_6_d = [[0.5, -0.1], [-0.2, 1.2], [0.8, 0.3], [1.8, 1.6]]

    solver.radial_basis_function_given_weights(input_layer_lgt_4_6_d, weights, centers_lgt_4_6, sigma,
                                               gaussian_hidden_function)

    # x_lgt_4_7 = np.linspace(0.05, 0.95, 19)
    x_lgt_4_7 = [[0.05], [0.2], [0.25], [0.3], [0.4], [0.43], [0.48], [0.6], [0.7], [0.8], [0.9], [0.95]]
    y_lgt_4_7 = [[0.0863, 0.2662, 0.2362, 0.1687, 0.1260, 0.1756, 0.3290, 0.6694, 0.4573, 0.3320, 0.4063, 0.3535]]
    centers1_lgt_4_7 = [[sum([0.05, 0.2, 0.25]) / 3], [sum([0.3, 0.4]) / 2], [sum([0.43, 0.48, 0.6, 0.7]) / 4],
                        [sum([0.8, 0.9, 0.95]) / 3]]
    solver.radial_basis_function(x_lgt_4_7, centers1_lgt_4_7, "avg", gaussian_hidden_function, y_lgt_4_7)

    # LGT 5_5
    x_lgt_5_5 = [[1, 0.5, 0.2, -1, -0.5, -0.2, 0.1, -0.1, 0],
                 [1, -1, 0.1, 0.5, -0.5, -0.1, 0.2, -0.2, 0],
                 [0.5, -0.5, -0.1, 0, -0.4, 0, 0.5, 0.5, 0.2],
                 [0.2, 1, -0.2, -1, -0.6, -0.1, 0.1, 0, 0.1]]
    x_shape = (3, 3)
    beta_lgt_5_5 = 0
    gamma_lgt_5_5 = 1
    epsilon_lgt_5_5 = 0.1
    result_shape = [[], [], [], []]
    solver.cnn_batch_normalisation(x_lgt_5_5, beta_lgt_5_5, gamma_lgt_5_5, epsilon_lgt_5_5, result_shape)
    X_lgt_5_6 = [[[0.2, 1, 0], [-1, 0, -0.1], [0.1, 0, 0.1]],
                 [[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]]]
    H_lgt_5_6 = [[[1, -.1], [1, -.1]], [[0.5, 0.5], [-0.5, -0.5]]]
    solver.cross_correlation_with_stride_dilation(X_lgt_5_6, H_lgt_5_6, padding=0)
    solver.cross_correlation_with_stride_dilation(X_lgt_5_6, H_lgt_5_6, padding=1)
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
    solver.convolution1X1(X_lgt_5_7, masks_lgt_5_7)

    # LGT 6 GAN
    real_x_lgt_6_4 = [[1, 2], [3, 4]]
    fake_x_lgt_6_4 = [[5, 6], [7, 8]]
    theta1_lgt_6_4 = 0.1
    theta2_lgt_6_4 = 0.2
    discriminator_lgt_6_4 = lambda x: 1 / (1 + math.exp(-(theta1_lgt_6_4 * x[0] - theta2_lgt_6_4 * x[1] - 2)))
    solver.gan_generator_and_discriminator(fake_x_lgt_6_4, real_x_lgt_6_4, discriminator_lgt_6_4)
    # lgt7 OJA RULE TEST
    # The KLT is in Matlab, not here of course
    X_t = [[5.0, 5.0, 4.4, 3.2],
           [6.2, 7.0, 6.3, 5.7],
           [5.5, 5.0, 5.2, 3.2],
           [3.1, 6.3, 4.0, 2.5],
           [6.2, 5.6, 2.3, 6.1]]
    oja_initial_weights = [-0.2, -0.2, 0.2, -0.0]
    learning_rate_oja = 0.01
    epoch_oja = 100
    x_lgt_7_6 = [[0, 1],
                 [3, 5],
                 [5, 4],
                 [5, 6],
                 [8, 7],
                 [9, 7]]

    learning_rate_lgt7_6 = 0.01
    oja_initial_weights_lgt7_6 = [-1, 0]
    solver.oja_rules(oja_initial_weights, X_t, learning_rate_oja, 1)

    w1 = [-1, 5]
    w2 = [2, -3]
    feature_vector_with_class = [[[1, 2], [2, 1], [3, 3]], [[6, 5], [7, 8]]]  # Look at the dimension of the array
    solver.fisher_method(feature_vector_with_class, w1)
    solver.fisher_method(feature_vector_with_class, w2)

    V_lgt_7_11 = [[-0.62, 0.44, -0.91],
                  [-0.81, -0.09, 0.02],
                  [0.74, -0.91, -0.60],
                  [-0.82, -0.92, 0.71],
                  [-0.26, 0.68, 0.15],
                  [0.80, -0.94, -0.83]]
    x_lgt_7_11 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_neuron_lgt_7_11 = [0, 0, 0, -1, 0, 0, 2]
    solver.extreme_learning_machine(V_lgt_7_11, x_lgt_7_11, output_neuron_lgt_7_11)

    x_lgt_7_12 = [-0.05, -0.95]
    V_lgt_7_12 = [[0.4, 0.55, 0.5, -0.1, -0.5, 0.9, 0.5, 0.45],
                  [-0.6, -0.45, -0.5, 0.9, -0.5, 0.1, 0.5, 0.55]]
    y_t1_lgt_7_12 = [[1, 0, 0, 0, 1, 0, 0, 0]]
    y_t2_lgt_7_12 = [[0, 0, 1, 0, 0, 0, -1, 0]]
    solver.sparse_coding(x_lgt_7_12, y_t1_lgt_7_12, V_lgt_7_12)
    solver.sparse_coding(x_lgt_7_12, y_t2_lgt_7_12, V_lgt_7_12)
    # LGT 8 SVM
    x_lgt_8_1_svm = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    class_x_lgt_8_1 = [1, 1, -1, -1]
    solver.svm(x_lgt_8_1_svm, class_x_lgt_8_1)

    x_lgt_8_2_svm = [[3, 1], [3, -1], [1, 0]]
    class_x_lgt_8_2 = [1, 1, -1]
    solver.svm(x_lgt_8_2_svm, class_x_lgt_8_2)

    x_lgt_8_3_svm = [[2, 2], [2, -2], [-2, -2], [-2, 2], [1, 1], [1, -1], [-1, -1], [-1, 1]]
    class_x_lgt_8_3 = [1, 1, 1, 1, -1, -1, -1, -1]


    def mapping_function(datum):
        if np.linalg.norm(datum) >= 2:
            new_datum = [4 - datum[1] / 2 + abs(datum[0] - datum[1]), 4 - datum[0] / 2 + abs(datum[0] - datum[1])]
        else:
            new_datum = [datum[0] - 2, datum[1] - 3]
        return new_datum


    new_data = solver.mapping_to_another_space(x_lgt_8_3_svm, mapping_function)
    print("The new data after mapping function is \n {}".format(new_data))
    # The new support vectors are [3, 3], [-1, -2]
    new_supporting_vector = [[3, 3], [-1, -2]]
    new_supporting_vector_class = [1, -1]
    solver.svm(new_supporting_vector, new_supporting_vector_class)
    # LGT 9

    x_lgt9_1 = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    label_lgt9_1 = [1, 1, -1, -1]
    weak_functions = [lambda x: 1 if x[0] > -0.5 else -1,
                      lambda x: -1 if x[0] > -0.5 else 1,
                      lambda x: 1 if x[0] > 0.5 else -1,
                      lambda x: -1 if x[0] > 0.5 else 1,
                      lambda x: 1 if x[1] > -0.5 else -1,
                      lambda x: -1 if x[1] > -0.5 else 1,
                      lambda x: 1 if x[1] > 0.5 else -1,
                      lambda x: -1 if x[1] > 0.5 else 1]
    solver.adaboost_algorithm(x_lgt9_1, label_lgt9_1, 3, weak_functions)
    sgn_lgt10_2 = lambda x: 1 if x >= 0 else -1
    solver.bagging(x_lgt9_1, label_lgt9_1, weak_functions, 0.5, sgn_lgt10_2)

    feature_vector_array = [[[-1, 3]], [[1, 2]], [[0, 1]], [[4, 0]], [[5, 4]], [[3, 2]]]
    x = solver.agglomerative_hierarchical_clustering(feature_vector_array, 4, distance_function=single_linkage)
    print(x)
    print("centroid_distance")
    feature_vector_array = [[[-1, 3]], [[1, 2]], [[0, 1]], [[4, 0]], [[5, 4]], [[3, 2]]]
    x = solver.agglomerative_hierarchical_clustering(feature_vector_array, 3, distance_function=centroid_distance)
    print(x)
