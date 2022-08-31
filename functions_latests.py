import numpy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import math
from scipy.signal import correlate2d
from collections import Counter
from numpy import linalg as LA


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


class solver():
    def __init__(self):
        pass

    # 2.1
    def dichotimizer(self, weight, bias, feature_vectors):
        print(
            "******************************************** The start of dichotimizer ********************************"
            "************")
        assert len(weight) == len(feature_vectors[0])
        label_hat_list = []
        for vector in feature_vectors:
            label_hat = bias
            for i in range(len(weight)):
                label_hat += weight[i] * vector[i]
            label_hat_tag = 1 if label_hat > 0 else 2
            label_hat_list.append({str(vector): {"value": label_hat, "label_hat": label_hat_tag}})
        for label in label_hat_list:
            print(label)
        print(
            "********************************************End of dichotimizer **************************************"
            "******\n\n\n")
        return label_hat_list

    # 2.2 augmented
    def augmented_dichotimiezer(self, a_t, y):
        print("****************************************Start of function augmented_dichotimiezer"
              "************************************************")
        """

        :param a: it's a, not transpose one.
        :param y:
        :return:
        """
        label_hat_list = []
        for vector in y:
            vector = np.transpose(vector)
            label_hat = np.matmul(a_t, vector)
            label_hat_tag = 1 if label_hat > 0 else 2
            label_hat_list.append({str(vector): {"value": label_hat, "label_hat": label_hat_tag}})
        for label in label_hat_list:
            print(label)
        print("************************************************ End of function augmented_dichotimiezer"
              "************************************************ \n\n\n")
        return label_hat_list

    # 2.6
    # Converges when no changes in parameters
    def batch_perceptron_learning_algorithm_with_augmentation_normalisation(self, x_list, a_t, learning_rate):
        print(
            "************************************************ "
            "batch_perceptron_learning_algorithm_with_augmentation_normalisation "
            "************************************************")
        """
        This algorithm is only for linear separably
        :param x_list:
        :param label:
        :param a_t:
        :param learning_rate:
        :return:
        """
        epoch = 0
        while True:
            old_a_t = a_t
            miss_match = []
            epoch += 1
            print("epoch {}".format(epoch))
            print("%15s%20s%50s" % ("y", "g(x) = a_t * y", "misclassified (g(x) < 0))"))
            for index, vector_t in enumerate(x_list):
                print("%15s" % vector_t, end="     |      ")
                vector = np.transpose(vector_t)
                label_hat = np.matmul(a_t, vector)
                print("%20s" % "+".join(
                    ["({} * {})".format(vector_t[i], a_t[i]) for i in range(len(vector_t))]),
                      end="")
                print("={}".format(np.matmul(a_t, vector)), end="")
                # since it's been sample normalised, g(x) < 0 is a miss classification
                if not label_hat > 0:  # a miss
                    # if it's a miss
                    miss_match.append(vector)
                    print("%20s" % "yes")
                else:
                    print("%20s" % "no")

            # in the end, update the a_t value
            print("Updated the a_t value", end=" ")
            a_t = np.transpose(np.transpose(a_t) + learning_rate * sum(miss_match))
            print(a_t)
            if np.array_equal(a_t, old_a_t):
                print("The result of the new parameter is {}".format(a_t))
                print("************************************************"
                      "End of solver.batch_perceptron_learning_algorithm_with_augmentation_normalisation"
                      "************************************************\n\n\n")
                return a_t

    def batch_perceptron_learning_algorithm_without_sn(self, a_t, x_list, label, learning_rate):
        print("************************************************"
              "start of function batch_perceptron_learning_algorithm_without_sn"
              "************************************************")
        score = 0
        epoch = 0
        while score != len(x_list):
            epoch += 1
            score = 0
            miss_match = []
            for index, vector_t in enumerate(x_list):
                vector = np.transpose(vector_t)
                label_hat = 1 if np.matmul(a_t, vector) > 0 else -1
                # since it's not been sample normalised, label_hat not equal = label is a mismatch
                if label_hat != label[index]:  # a miss
                    # if it's a miss
                    miss_match.append(label[index] * np.array(vector))
                else:
                    score += 1
            # in the end, update the a_t value
            a_t = np.transpose(np.transpose(a_t) + learning_rate * sum(miss_match))
        print("************************************************ "
              "End of batch_perceptron_learning_algorithm_without_sn"
              "************************************************\n\n\n")
        return a_t

    # By default, it's been normalised
    # Stop when there are no changes in the new parameter
    # The parameters converged;
    def sequential_delta_learning_algorithm_stop_with_augmentation_normalisation(self, x_list, a_t, learning_rate):
        """
        label_hat = a_t * y
        a_t = a_t + learning_rate * yk where yk is wrongly label feature.
        :param a_t: augmented variable a_t = [w_0 wT]_T
        :param y: augmented variable y = [1 xT]_T
        :param learning_rate:
        :return:
        """
        print("************************************************"
              "start of function sequential_delta_learning_algorithm_stop_with_augmentation_normalisation"
              "************************************************")
        epoch = 0
        new_a_t = a_t
        while True:
            old_a_t = a_t
            epoch += 1
            print("epoch {}".format(epoch))
            print("%15s%60s%40s" % ("y", "g(x) = a_t * y", "updated a_t"))
            for index, vector_t in enumerate(x_list):
                print("%15s" % (vector_t), end="")
                vector = np.transpose(vector_t)
                label_hat = np.matmul(a_t, vector)
                print("%60s" % "+".join(
                    ["({} * {})".format(vector_t[i], a_t[i]) for i in range(len(vector_t))]),
                      end=" = ")
                print(label_hat, end="")
                # since it's been sample normalised, g(x) < 0 is a miss classification
                if not label_hat > 0:  # a miss
                    # since it's a sequential gradient descent, the a_t changes immediately
                    print("%40s" % (str(a_t) + "+" + str(learning_rate) + "*" + str(vector) + "="), end="")
                    a_t += np.transpose(learning_rate * vector)
                    print(a_t)
                else:
                    print("%40s" % (a_t))
                new_a_t = a_t
            if np.array_equal(new_a_t, old_a_t):
                print("The final parameter is {}".format(new_a_t))
                print("************************************************"
                      "End of sequential_delta_learning_algorithm_stop_with_augmentation_normalisation"
                      "************************************************\n\n\n")
                return new_a_t

    # Different from the previous one, stop when no mis-classification
    def sequential_delta_learning_algorithm_stop_with_augmentation_normalisation_2(self, x_list, a_t, learning_rate):
        print("************************************************"
              "start of sequential_delta_learning_algorithm_stop_with_augmentation_normalisation_2"
              "************************************************")
        epoch = 0
        score = 0
        while score != len(x_list):
            score = 0
            epoch += 1
            print("epoch {}".format(epoch))
            print("%15s%60s%40s" % ("y", "g(x) = a_t * y", "updated a_t"))
            for index, vector_t in enumerate(x_list):
                print("%15s" % (vector_t), end="")
                vector = np.transpose(vector_t)
                label_hat = np.matmul(a_t, vector)
                print("%60s" % "+".join(
                    ["({} * {})".format(vector_t[i], a_t[i]) for i in range(len(vector_t))]),
                      end=" = ")
                print(label_hat, end="")
                # since it's been sample normalised, g(x) < 0 is a miss classification
                if not label_hat > 0:  # a miss
                    # since it's a sequential gradient descent, the a_t changes immediately
                    print("%40s" % (str(a_t) + "+" + str(learning_rate) + "*" + str(vector) + "="), end="")
                    a_t += np.transpose(learning_rate * vector)
                    print(a_t)
                else:
                    score += 1
                    print("%40s" % (a_t))
            print("The final parameter is {}".format(a_t))
            print("************************************************ "
                  "End of sequential_delta_learning_algorithm_stop_with_augmentation_normalisation_2"
                  "************************************************\n\n\n")
            return a_t

    # When it's not sample normalised
    # TODO: Instructions
    def sequential_perceptron_learning_algorithm_without_normalisation(self, x_list, a_t, label, learning_rate=1):
        """
        label_hat = a_t * y
        a_t = a_t + learning_rate * yk where yk is wrongly label feature.
        :param a_t: augmented variable a_t = [w_0 wT]_T
        :param y: augmented variable y = [1 xT]_T
        :param learning_rate:
        :return:
        """
        print("%5s%20s%20s%20s%5s%25s" % ("iteration", "a_t_old", "y_t", "g(x)=a_t * y", "w_k", "at_new=at_old+wk*yt"))
        new_t = a_t
        iteration = 0
        while True:
            old_a_t = a_t
            score = 0
            for index, vector_t in enumerate(x_list):
                iteration += 1
                vector = np.transpose(vector_t)
                label_hat = 1 if np.matmul(a_t, vector) > 0 else -1
                print("%5s" % iteration, end="")
                print("%20s" % a_t, end="")
                print("%20s" % vector_t, end="")
                print("%20s" % np.matmul(a_t, vector), end="")
                print("%10s" % label[index], end="")
                # since it's been sample normalised, g(x) < 0 is a miss classification
                if label_hat != label[index]:  # a miss
                    # since it's a sequential gradient descent, the a_t changes immediately
                    a_t += np.transpose(learning_rate * vector) * label[index]
                print("%20s" % a_t)
                new_t = a_t

            if np.array_equal(new_t, old_a_t):
                print("The final result is {}".format(new_t))
                print("End of sequential_perceptron_learning_algorithm_without_normalisation \n\n\n")
                return a_t

    def heaviside_func(self, value, threshold=1e-15):
        if value > threshold:
            return 1
        else:
            return 0

    def check_all_match(self, w, x, label):
        assert len(label) == len(x)
        for index, vector_t in enumerate(x):
            label_hat = np.matmul(w, np.transpose(vector_t))
            print(label_hat)
            if label[index] != self.heaviside_func(label_hat):
                return False
        return True

    def row_wise_normalised(self, ori_matrix):
        return normalize(ori_matrix, norm="l1")

    # # stop when no changes
    # # TODO: BUG NOT STOPPING
    # def sequential_delta_learning_algorithm(self, x, initial_a_t, label, learning_rate=1):
    #     """
    #     in this algorithm, parameters must be augmented first.
    #     :param w:
    #     :param label:
    #     :param x:
    #     :return:
    #     """
    #     w = np.array(initial_a_t)
    #     x = np.array(x)
    #     print(len(label), len(x))
    #     assert len(label) == len(x)
    #     while True:
    #         old_w = w
    #         for index, vector_t in enumerate(x):
    #             vector = vector_t.transpose()
    #             new_w = w + learning_rate * (label[index] - self.heaviside_func(np.matmul(w, vector))) * np.array(
    #                 vector_t)
    #             w = new_w
    #         print(w)
    #         if np.array_equal(new_w, old_w):
    #             return w

    # Using in Linear Threshold Unit
    def batch_delta_learning_rule(self, x_list, w, label, learning_rate=1):
        epoch = 0
        while True:
            old_w = w
            epoch += 1
            print("epoch {}".format(epoch))
            print("initial w {}".format(w))
            pool = []
            print("%5s%5s%40s%12s%20s%10s" % ("x_t", "t", "y=H(wx)", "t-y", "n(t-y)x_t", "w"))
            for index, vector_t in enumerate(x_list):
                vector = np.transpose(vector_t)
                trans_value = np.matmul(w, vector)
                label_hat = self.heaviside_func(trans_value)
                # since it's not been sample normalised, label_hat not equal = label is a mismatch
                print("%5s" % (vector_t), end="")
                print("%5s" % (label[index]), end="")
                if len(w) == 1:
                    print("%30s" % (
                            "H(" + "+".join(["({} * {})".format(vector_t[i], w[0][i]) for i in range(len(vector_t))])),
                          end=")=")
                else:
                    print("%30s" % (
                            "H(" + "+".join(["({} * {})".format(vector_t[i], w[i]) for i in range(len(vector_t))])),
                          end=")=")
                trans_value = np.matmul(w, vector)
                print(("H({})").format(trans_value), end="=")
                label_hat = self.heaviside_func(trans_value)
                print(label_hat, end="")
                dif = label[index] - label_hat
                print("%10s" % (str(label[index]) + "-" + str(label_hat)), end="")
                print("%15s" % (str(learning_rate) + '*' + "(" + str(label[index]) + "-" + str(label_hat) + ")"))
                if label_hat != label[index]:
                    pool.append(learning_rate * (label[index] - label_hat) * np.array(vector_t))
                else:
                    pool.append(learning_rate * (label[index] - label_hat) * np.array(vector_t))

            print("total weight change = %s" % sum(pool))
            # in the end, update the a_t value
            w = w + learning_rate * sum(pool)
            print("after updating, w is {}".format(w))
            if np.array_equal(w, old_w):
                print("So the final theta is {} and w is {}".format(-w[0], w[1:]))
                print("End of batch_delta_learning_rule\n\n\n ")
                return w

    def knn(self, X, new_item, k):
        new_item = np.array(new_item)
        distance_dict = []
        for item in X:
            value = np.array(item["value"])
            distance_dict.append({"class": item["class"], "distance": self.euclidean_distance(value, new_item)})
        print(distance_dict)
        candidate = sorted(distance_dict, key=lambda d: d['distance'])[:k]
        print(candidate)
        candidate_class = [item["class"] for item in candidate]
        mode_of_candidate = max(set(candidate_class), key=candidate_class.count)
        print(Counter(candidate_class))
        print("The corresponding predicted class should be *{}*".format(mode_of_candidate))
        print("End of KNN \n\n\n")

        return mode_of_candidate

    def pseudo_inverse_with_margin(self, y, margin_t):
        """
        Y*a = margin, where a is unknown
        a = (Y_t*Y)^-1*Y_t * margin
        :param y: y_t are data after sample normalisation
        :param margin_t: Just a vector not N*1
        :return:
        """
        print("************************************************"
              " start of pseudo_inverse_with_margin"
              "************************************************")
        margin_t = np.array(margin_t)
        margin = margin_t.T
        pseudo_inverse_matrix = np.linalg.pinv(y)
        print("The Pseudo-inverse matrix of y is \n {}".format(pseudo_inverse_matrix))
        a = pseudo_inverse_matrix @ margin
        print("The parameter is\n {}".format(np.round(a, 4)))
        print("************************************************ "
              "End of pseudo_inverse_with_margin "
              "************************************************\n\n\n")
        return a

    # quiz 2.1
    # lgt 2_14
    def sequential_widrow_hoff_learning(self, initial_a_t, margin_t, feature_vector_t, learning_rate, iteration):
        print("************************************************ "
              "start of sequential_widrow_hoff_learning"
              "************************************************ ")
        a_t = np.array(initial_a_t)
        margin = np.array(margin_t).T
        feature_vector_t = np.array(feature_vector_t)
        print("%5s%60s%100s" % ("iteration", "a_Tyk", "aT_new = a_T+n(bk - aTyk)yTk"))
        epoch = 0
        while True:
            for i in range(len(feature_vector_t)):
                epoch += 1
                feature_vector = feature_vector_t[i].transpose()
                print("%5s" % (epoch), end="")
                a_Tyk = np.round((a_t @ feature_vector), 4)
                print("%60s" % ("{} @ {} = {}".format(a_t, feature_vector, a_Tyk)), end="")

                aT_new = np.round(a_t + learning_rate * (margin[i] - a_Tyk) * feature_vector.transpose(), 4)
                print("%100s" % ("{} = {} + {}").format("(" + ",".join(["%.4f" % i for i in aT_new[0]]) + ")",
                                                        "(" + ",".join(["%.4f" % i for i in a_t[0]]) + ")",
                                                        "(" + ",".join(["%.4f" % i for i in (learning_rate * (
                                                                margin[i] - a_Tyk) * feature_vector.transpose())]) + ")"
                                                        ))
                a_t = aT_new
                if epoch >= iteration:
                    print("************************************************  "
                          "End of sequential_Widrow_hoff_learning "
                          "************************************************ \n\n\n")
                    return a_t

    def euclidean_distance(self, feature1, feature2):
        return np.linalg.norm(feature1 - feature2)

    ############################ LGT3 #############################
    """
        In the NN, the theta needs negated
    """

    def nn_given_wights_and_input(self, x_t, weights):
        """
        feature first
        weights next
        :param x_t:
        :param weights:
        :return:
        """
        print("************************************************ "
              "nn_given_wights_and_input"
              "************************************************ ")
        x = np.array(x_t).T
        linear_weighted_sum = weights @ x
        print("H({}) = {}".format(linear_weighted_sum, np.round(self.heaviside_func(linear_weighted_sum), 4)))
        print("************************************************  "
              "End of nn_given_wights_and_input"
              "************************************************ \n\n\n")
        return self.heaviside_func(linear_weighted_sum)

        # 3 Sequential Delta Learning Algorithm this is for Linear Threshold Units

    # Kimhom Question:
    # According to Page 20 iteration stops when w does not change
    # but in the very next page, the condition when iteration stops becomes all match
    def sequential_delta_learning_rule(self, x, w, label, learning_rate):
        """
        in this algorithm, parameters must be augmented first.
        Converge when all the items correctly classified
        :param w:
        :param label:
        :param x:
        :return:
        """
        print("************************************************ "
              "sequential_delta_learning_rule"
              "************************************************ ")
        w = np.array(w)
        x = np.array(x)
        print(len(label), len(x))
        assert len(label) == len(x)
        epoch = 0
        score = 0
        while True:
            epoch += 1
            print("epoch {}".format(epoch))
            print("%5s%5s%40s%20s%20s%20s" % ("x_t", "t", "y=H(wx)", "t-y", "n(t-y)x_t", "w"))
            for index, vector_t in enumerate(x):
                old_w = w
                vector = vector_t.transpose()
                print("%5s" % (vector_t), end="")
                print("%5s   " % (label[index]), end="")
                if len(w) == 1:
                    print("%30s" % (
                            "H(" + "+".join(["({} * {})".format(vector_t[i], w[0][i]) for i in range(len(vector_t))])),
                          end=")=")
                else:
                    print("%30s" % (
                            "H(" + "+".join(["({} * {})".format(vector_t[i], w[i]) for i in range(len(vector_t))])),
                          end=")=")
                trans_value = np.matmul(w, vector)
                print(("H({})").format(trans_value), end="=")
                label_hat = self.heaviside_func(trans_value)
                print(label_hat, end="")
                dif = label[index] - label_hat
                print("%10s" % (str(label[index]) + "-" + str(label_hat)), end="")
                new_w = w + learning_rate * dif * np.array(vector_t)
                print("%30s" % (str(learning_rate) + "*" + str(dif) + "*" + str(vector_t) + '=' + str(
                    learning_rate * dif * np.array(vector_t))), end="  ")
                print("%5s" % (str(new_w)))
                w = new_w
                if dif == 0:
                    score += 1
                else:
                    score = 0
                if score == len(label):
                    print("The FINAL w is {}".format(w))
                    print("So the final theta is {} and w is {}".format(-w[0], w[1:]))
                    print(" ************************************************ "
                          "The end of sequential_delta_learning_rule"
                          "************************************************  \n\n\n")

                    return w

    # 3.9
    # TODO: BUG OF NUMPY
    def regulatory_feedback(self, weights, x_t, y_t, sig_1, sig_2, iteration=5):
        print("************************************************ "
              "regulatory_feedback"
              "************************************************ ")
        weights = np.round(np.array(weights), 4)
        nor_w = self.row_wise_normalised(weights)
        x_t = np.round(np.array(x_t), 4)
        weights_t = np.transpose(np.round(np.array(weights), 4))
        x = np.transpose(x_t)
        y = np.transpose(y_t)
        for i in range(iteration):
            print("\nepoch {}\n".format(i + 1))
            max_wy = weights_t @ y
            max_wy[max_wy < [sig_2]] = [.9]
            e = np.round(np.divide(x, max_wy), 4)
            y[y < [sig_1]] = sig_1
            y = np.round(np.multiply(y, nor_w @ e), 4)
            print("e = {} \n,y = {}\n".format(e, y))
        print("Finally the Y is {}".format(y))
        print("************************************************ "
              "regulatory_feedback"
              "************************************************ \n\n\n")
        return y

    # 3.7, 3.8
    def negative_feedback_neural_network_update_e_only(self, weights, update_rate, x_t, y_t, iteration):
        print("************************************************ "
              "negative_feedback_neural_network_update_e_only"
              "************************************************ ")
        weights = np.round(np.array(weights), 4)
        x_t = np.round(np.array(x_t), 4)
        weights_t = np.transpose(np.round(np.array(weights), 4))
        x = np.transpose(x_t)
        y = np.transpose(y_t)
        print("%5s%20s%20s%20s%30s" % ("iteration", "e_T", "(We)T", "y_T", "(W_Ty)T"))
        for _ in range(iteration):
            print("%5s" % (_ + 1), end="")
            e = np.round(x - weights_t @ y, 4)
            print("%30s" % (e.transpose()), end="")
            print("%20s" % ((weights @ e).transpose()), end="")
            y = np.round(y + update_rate * weights @ e, 4)
            print("%20s" % (y.transpose()), end="")
            print("%30s" % (weights_t @ y).transpose())
            # print("e = {} \n,y = {}\n".format(e, y))
        print("Finally the Y_t is {}".format(y))
        print("************************************************ "
              "End of negative_feedback_neural_network_update_e_only"
              "************************************************  \n\n\n")
        return y

    #
    # """
    # *************************************** LGT4 ******************************************
    # """
    #
    def NN_given_weights(self, input_layer_t, hidden_layers, hidden_layer_bias, activation_function):
        """

        :param input_layer:
        :param hidden_layers:
        :param hidden_layer_bias:
        :param activation_function: A linear function returns the input itself,
        :return:
        """
        print("************************************************ "
              "NN_given_weights"
              "************************************************ ")
        input_layer_t = activation_function[0](input_layer_t)
        input_layer = np.array(input_layer_t).T
        for index, hidden_layer in enumerate(hidden_layers):
            temp_output = hidden_layer @ input_layer + np.array(hidden_layer_bias[index]).T
            after_activation = activation_function[index + 1](temp_output)
            input_layer = after_activation

        print("The result would be \n{}".format(input_layer))
        print("************************************************ "
              "end of NN_given_weights "
              "************************************************ \n\n\n")

    def radial_basis_function(self, input_layer, centers, sigma_way, hidden_function, label_t):
        print(
            "****************************************Start radial_basis_function**************************************")
        # There are 3 layers in total.
        print(centers)
        label = np.array(label_t).T
        nh = len(centers)
        if sigma_way == "max":
            p = 0
            for index_i in range(len(centers)):
                for index_j in range(index_i + 1, len(centers)):
                    distance = self.l2_norm(centers[index_i], centers[index_j])
                    if distance > p:
                        p = distance
            sigma = p / math.sqrt(2 * nh)
        elif sigma_way == "avg":
            total_norm_distance = 0
            times = 0
            for index_i in range(len(centers)):
                for index_j in range(index_i + 1, len(centers)):
                    times += 1
                    total_norm_distance += self.l2_norm(centers[index_i], centers[index_j])
            sigma = 2 * (total_norm_distance / times)
        hidden_layer_outputs = []
        for item in input_layer:
            temp_hidden_output = []
            for center in centers:
                distance = self.l2_norm(center, item)
                # Hidden function should be used here.
                temp_hidden_output.append(hidden_function(distance, sigma))
            hidden_layer_outputs.append(np.round(temp_hidden_output, 6))
        print("Hidden Layer Output \n {}".format(hidden_layer_outputs))
        # Since we get the hidden layer output, given formula, (where we use X to represent the hidden layer,
        # W to represent the weights and t the label )
        # Wx = t, where x needs to be augmented
        # Augmenting X (The tail is the intercept, rather than the head. )
        augmented_x = [np.insert(item, len(item), 1) for item in hidden_layer_outputs]
        weights = np.linalg.pinv(augmented_x) @ label
        print(
            " *********radial_basis_function*************************************** "
            "The final weights of would be ***********************************************"
            " \n {} \n where the last item of the weight is the intercept"
            ""
            "\n\n\n".format(weights))
        print("w0 = {} \n, w=\n{}".format(weights[-1], weights[:-1]))
        return weights, sigma

    def radial_basis_function_given_weights(self, input_layers, weights, centers, sigma, hidden_function):
        print(" ************************************************ "
              "radial_basis_function_given_weights"
              " ************************************************ ")
        hidden_layer_outputs = []
        for item in input_layers:
            temp_hidden_output = []
            for center in centers:
                distance = self.l2_norm(center, item)
                # Hidden function should be used here.
                temp_hidden_output.append(hidden_function(distance, sigma))
            hidden_layer_outputs.append(np.round(temp_hidden_output, 6))
        augmented_x = [np.insert(item, len(item), 1) for item in hidden_layer_outputs]
        print(augmented_x)
        label = augmented_x @ weights
        flat_label = [1 if item > 0.5 else 0 for sublist in label for item in sublist]

        print(" The raw_labels of the input are \n {}".format(flat_label))
        print(" ************************************************ "
              "End of radial_basis_function_given_weights "
              " ************************************************ \n\n\n")

    # LGT4_7
    # pending
    def sketch_function(self, x, y):
        from matplotlib import pyplot as plt
        plt.plot(x, y)
        plt.show()
    # pending
    def performance_matrix(self, parameters, feature_vectors, labels):
        print(" ************************************************ "
              "performance_matrix"
              " ************************************************ ")
        parameters = np.array(parameters)
        feature_vectors = np.array(feature_vectors)
        confusion_matrix = {"true positive": 0, "false positive": 0, "true negative": 0, "false negative": 0}
        for index, vector_t in enumerate(feature_vectors):
            vector = vector_t.transpose()
            label_hat = parameters @ vector
            label_hat = self.heaviside_func(label_hat)
            if label_hat != labels[index]:
                if labels[index] == 1:
                    confusion_matrix["false negative"] += 1
                else:
                    confusion_matrix["false positive"] += 1
            else:
                if labels[index] == 1:
                    confusion_matrix["true positive"] += 1
                else:
                    confusion_matrix["true negative"] += 1
        print("Precession {}".format(confusion_matrix["true positive"] / (
                confusion_matrix["true positive"] + confusion_matrix["false positive"])))
        print("Recall {}".format(confusion_matrix["true positive"] / (
                confusion_matrix["true positive"] + confusion_matrix["false negative"])))
        print(" ************************************************ "
              "performance_matrix"
              " ************************************************ \n\n\n")
        return confusion_matrix

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def l2_norm(self, vector_1, vector_2):
        assert len(vector_1) == len(vector_2)
        # TO make this numpy array
        vector_1 = np.array(vector_1)
        vector_2 = np.array(vector_2)
        l2 = np.power(np.sum(np.power((vector_1 - vector_2), 2)), 0.5)
        return l2

    # pending
    def fixed_centers_selection_RBF(self, X, centers):
        print(" ************************************************ "
              "fixed_centers_selection_RBF"
              " ************************************************ ")

        def gaussian(sigma, dj_matrix):
            dj_matrix = np.array(dj_matrix)
            print("result")
            activations = np.exp(- np.power(dj_matrix, 2) / (2 * (sigma ** 2)))
            return np.round(activations, 4)

        hidden_layer = []
        for x in X:
            net = []
            for center in centers:
                net.append(self.l2_norm(center, x))
            hidden_layer.append(net)
        print(hidden_layer)
        print(np.max(hidden_layer))
        sigma = np.max(hidden_layer) / (math.pow(2 * len(centers), 1 / 2))
        print(sigma)
        print("hidden layer after activation")
        activations_matrix = gaussian(sigma, hidden_layer)
        print(activations_matrix)
        print(" ************************************************ "
              "fixed_centers_selection_RBF"
              " ************************************************\n\n\n ")

    # """
    #     ******************************** LGT5 ****************************************
    # """

    def relu(self, net):
        """

        :param net: where net is not a scalar.
        :return:
        """
        net = np.array(net)
        net[net < 0] = 0
        print("The matrix after being processed by ReLu is \n {}\n\n\n".format(net))
        return net

    def lrelu(self, net, alpha):
        net = np.array(net)
        net[net < 0] = net[net < 0] * alpha
        print("The matrix after being processed by LreLu is \n {} \nwhen alpha is {}\n\n\n".format(net, alpha))
        return net

    def tanh(self, net):
        net = np.round(np.tanh(net), 4)
        print("The matrix after being processed by tanh is \n {}\n\n\n".format(net))
        return net

    def heaviside_matrix(self, net, threshold):
        print("heaviside matrix")
        net = np.array(net)
        net[net > threshold] = 1
        net[net == threshold] = 0.5
        net[net < threshold] = 0
        print("The matrix after being processed by LreLu is \n {} \nwhen alpha is {}\n\n\n".format(net, threshold))
        return net

    def variance_of_matrix(self, matrix):
        matrix_mean = np.mean(matrix)
        variance_matrix = np.power(matrix - matrix_mean, 2)
        return np.sum(variance_matrix)

    def cnn_batch_normalisation(self, X, beta, gamma, epsilon, result):
        print(" ************************************************ "
              "CNN batch normalisation"
              " ************************************************ ")
        X = np.array(X)
        # We do this position by position
        for index in range(len(X[0])):
            # We first need to know the Expectation of every position
            summation = 0
            for item in X:
                summation += item[index]
            expectation = summation / len(X)
            var_summation = 0
            for item in X:
                var_summation += (item[index] - expectation) ** 2
            var = var_summation / 4
            for sub_index, item in enumerate(X):
                result[sub_index].append(beta + gamma * (item[index] - expectation) / (math.sqrt(var + epsilon)))
        for index in range(len(result)):
            result[index] = np.array(result[index])
            result[index] = np.round(result[index].reshape((3, 3)), 4)
            print(result[index])
        print(" ************************************************  "
              "End of CNN batch normalisation"
              " ************************************************ \n\n\n")

    # TODO:
    # Will not be used this time
    def cross_correlation_with_stride_dilation(self, inputs, H, padding=1, stride=1, dilation=1):
        # taking the sum,f
        # wrongly Kim comments
        print("padding = {}, stride = {}, dilation = {}".format(padding, stride, dilation))
        print("cross_correlation_with_stride_dilation")
        # padding process
        inputs = np.array(inputs)
        H = np.array(H)
        output = []
        for index, x in enumerate(inputs):
            if padding:
                x = np.pad(x, padding)
            output.append(correlate2d(x, H[index], "valid"))
        print("The step before summation is\n {}".format(output))
        print("The actual output will be \n {}".format(np.sum(output, axis=0)))
        print("End of cross_correlation_with_stride_dilation \n\n\n")
        return output

    def convolution1X1(self, inputs, masks):
        print(" ************************************************ "
              "convolution1X1"
              " ************************************************ ")
        assert len(inputs) == len(masks)
        inputs = np.array(inputs)
        sum_array = []
        for index, mask in enumerate(masks):
            sum_array.append(mask * inputs[index])
        result = np.sum(sum_array, axis=0)
        print(" ************************************************ "
              "The Output of the Convolution would be \n {}"
              " ************************************************ ".format(result))
        return result

    def hebbian_learning_rule(self, w, x_features, learning_rate, epoch):
        print(" ************************************************ "
              "hebbian_learning_rule"
              " ************************************************ ")
        x_features = np.array(x_features)
        for i in range(epoch):
            for x in x_features:
                y = np.sum(np.multiply(w, x))
                delta_w = learning_rate * y * x
                w = w + delta_w
        print(" ************************************************ "
              "hebbian_learning_rule"
              " ************************************************ \n\n\n")
        return w

    def oja_rules(self, w, x_features, learning_rate, epoch):
        print(" ************************************************ "
              "oja_rules"
              " ************************************************ ")
        # in Oja's rule, and Hebbian rule, they both use the zero-mean to calculate the weights.
        x_features = np.array(x_features)
        mean_vector = x_features.mean(axis=0)
        print(mean_vector)
        zero_mean_data = x_features - mean_vector
        w = np.array(w)
        for i in range(epoch):
            delta_w = 0
            print("%40s%20s%40s%40s%10s" % ("X_t(zero_mean)", "y = wx", "x_t-yw", "ny(x_t-yw)", "w"))
            for x in zero_mean_data:
                print('%40s' % (x), end="")
                y = np.sum(np.multiply(w, x))
                print("%20s" % (round(y, 2)), end="")
                print("%40s" % np.round((x - y * w), 4), end="")
                print("%40s" % np.round((learning_rate * y * (x - y * w)), 4))
                delta_w += learning_rate * y * (x - y * w)
            print("\nTotal weight change {}".format(delta_w))
            w = w + delta_w
            print("epoch {}, initial w = {}".format(i + 1, w))
        print(" ************************************************ "
              "oja_rules"
              " ************************************************ \n\n\n")
        return w

    def kl_transform(self, x_features, principal_numbers: int):
        print(" ************************************************ "
              "KLT"
              " ************************************************ ")
        mean_vector = [np.sum(x_features, axis=0) / len(x_features)]
        mean_vector = np.array(mean_vector).transpose()
        print("mean vector:\n", mean_vector)
        C = []  # a covariance matrix
        for x_t in x_features:
            x = np.transpose(np.array([x_t]))
            C.append((x - mean_vector) @ (x - mean_vector).transpose())
        C = [np.sum(C, axis=0) / len(x_features)]
        print("Covariance matrix\n", C)
        e_values, e_vectors = LA.eig(np.array(C))

        e_values = e_values[0]
        e_vectors = e_vectors[0]
        # Order eigenvalues from large to small, and discard small eigenvalues and their respective vectors
        print(e_values)
        print("e_vectors \n")
        print(e_vectors)
        sorted_e_values, sorted_e_vectors = zip(*sorted(zip(e_values, e_vectors), reverse=True))
        print(sorted_e_vectors)
        e_vectors_hat = sorted_e_vectors[:principal_numbers]
        print("e_vectors_hat \n")
        print(e_vectors_hat)
        y_all = []

        for index, x_t in enumerate(x_features):
            x = np.transpose(np.array([x_t]))
            y = e_vectors_hat @ (x - mean_vector)
            print("y{} = ".format(index), "e_vectors_hat", e_vectors_hat, "(x - mean_vector)", (x - mean_vector), "y",
                  y)
            y_all.append(y)

        # The proportion of the variance is given by the sum of eigenvalues for selected components divided by
        # the sum of all eigenvalues
        proportion = sum(sorted_e_values[:principal_numbers]) / sum(sorted_e_values)
        # This could indicate the information loss rate.
        print("Proportion of principal numbers {} is {}".format(principal_numbers, proportion))
        print(" ************************************************ "
              "KLT"
              " ************************************************ \n\n\n")
        return y_all

    def fisher_method(self, feature_with_class, w):
        print("****************************************************************Start of Fisher Method "
              "****************************************************************")
        w = np.array(w)
        # feature_with_class = np.array(feature_with_class)
        mean_vector = []
        si_list = []
        for index, value in enumerate(feature_with_class):
            within_mean_vector = np.array(np.sum(value, axis=0) / len(value))
            mean_vector.append(within_mean_vector)
            si_value = 0
            for item in value:
                step_si_value = (np.sum(w * (item - within_mean_vector))) ** 2
                si_value += step_si_value
            si_list.append(si_value)
        sb = (np.sum(w * (mean_vector[0] - mean_vector[1]))) ** 2
        sw = np.sum(si_list)
        print("SB: ", sb)
        print("SW: ", sw)
        print("SB / SW = {} / {} = {}".format(sb, sw, sb / sw))
        print("****************************************************************End of Fisher Method "
              "****************************************************************")
        return sb / sw

    # LGT 6
    # Discriminator function
    def gan_generator_and_discriminator(self, fake_dateset, real_dataset, discriminator_function):
        # The expectation for sample is (element_value * element_probability)
        # By default the probability is uniform.
        print("****************************************************************"
              "gan_generator_and_discriminator"
              "****************************************************************")
        expectation_x_pdatax = 0  # this one is for real elements
        expectation_z_pz = 0  # This is one is for fake elements
        for index in range(len(real_dataset)):
            expectation_x_pdatax += math.log(discriminator_function(real_dataset[index])) * (1 / len(real_dataset))
            expectation_z_pz += math.log((1 - discriminator_function(fake_dateset[index]))) * (1 / len(real_dataset))
        print(expectation_x_pdatax, expectation_z_pz)
        print("Ex~pdata(x)[logD(x)] = {}\n "
              "Ez~pz(z)[ln(1 - D(G(z)))] = {}".format(expectation_x_pdatax, expectation_z_pz))
        print("V(D ,G) = {}".format(expectation_x_pdatax + expectation_z_pz))
        print("****************************************************************"
              "gan_generator_and_discriminator"
              "****************************************************************\n\n\n")

    # LGT 7
    # The extreme learning machine
    def extreme_learning_machine(self, random_matrix, x_features, output_neuron):
        print("****************************************************************"
              "extreme_learning_machine"
              "****************************************************************")
        random_matrix = np.array(random_matrix)
        x_features = np.array(x_features)
        x_features = x_features.transpose()
        x_features = np.insert(x_features, 0, 1, axis=0)
        VX = random_matrix @ x_features
        print("VX\n", VX)
        Y = self.heaviside_matrix(VX, 0)
        print("Y\n", Y)
        Y = np.insert(Y, 0, 1, axis=0)
        Z = output_neuron @ Y
        print("The final result is\n", Z)
        print("****************************************************************"
              "extreme_learning_machine"
              "****************************************************************")
        return Z

    def sparse_coding(self, original_x, y_t, v_t):
        print("****************************************************************"
              "sparse_coding (The smaller the better)"
              "****************************************************************")
        print("If the error is the same, we choose the sparser one. "
              "By sparser, we mean less NON-zero element.")
        # y = g(Vx), such that y contains only a few non-zero elements where V is a matrix of weights
        from numpy.linalg import norm
        from math import sqrt
        v_t = np.array(v_t)
        y_t = np.array(y_t)
        y = y_t.transpose()
        x = v_t @ y
        print("Vt@y1 \n", x)
        error = norm(original_x - x.T)
        print("||{} - {} ||2 = sqrt({} + {}) = {}".format(original_x, x.T, (original_x - x.T), (original_x - x.T),
                                                          error))
        # smaller the error, better the performance!
        print("****************************************************************"
              "sparse_coding"
              "**************************************************************** \n\n\n")
        return error

    def equation_solver(self, parameters, results):
        inverse_parameters = np.linalg.pinv(np.array(parameters))
        results = np.array(results)
        unknowns = inverse_parameters @ results.T
        return unknowns

    def kmeans(self, data_set, initial_center):

        ite = 0
        while True:
            ite += 1
            result = [[] for i in range(len(initial_center))]
            for data in data_set:
                min_distance = float("inf")
                min_index = 0
                for index, center in enumerate(initial_center):
                    dist = np.linalg.norm(np.array(data) - np.array(center))
                    if dist < min_distance:
                        min_distance = dist
                        min_index = index
                result[min_index].append(data)
            new_center = np.mean(result, axis=1)
            if np.all(new_center == initial_center):
                break
            initial_center = new_center
        return result, initial_center

    def pca(self, dataset, features_num, new_data=[]):
        from sys import getsizeof
        pca = PCA(n_components=features_num)
        principalComponents = pca.fit_transform(dataset)
        cov_matrix = np.dot(np.array(dataset).T, np.array(dataset)) / 3
        print(principalComponents)
        if len(new_data):
            print(pca.transform(new_data))
        print(pca.explained_variance_)
        print(cov_matrix)

    def sample_normalised(self, labels, dataset):
        normalized_dataset = []
        for index, label in enumerate(labels):
            temp = np.concatenate(([1], dataset[index]), axis=0)
            if label == 1:
                normalized_dataset.append(np.array(temp))
            else:
                normalized_dataset.append(-1 * np.array(temp))
        return np.array(normalized_dataset)

    def compute_euclidean_diff(self, array1, array2):
        """
        """
        return np.sqrt(np.sum(np.square(np.absolute(array1 - array2))))

    def svm(self, supporting_vectors, classes):
        """
        Only for 2 dimensions
        :param supporting_vectors:
        :param classes:
        :return:
        """
        print("****************************************************************Start of Function SVM"
              "****************************************************************")
        assert len(supporting_vectors) == len(classes)
        # build parameter matrix
        print([[[item[0], classes[index]] for index, item in enumerate(supporting_vectors)],
               [[item[1], classes[index]] for index, item in enumerate(supporting_vectors)]])
        w_matrix = np.array([[item[0] * classes[index] for index, item in enumerate(supporting_vectors)],
                             [item[1] * classes[index] for index, item in enumerate(supporting_vectors)]])

        w_symbol = ["+".join([str(num) + "λ" + str(index + 1) for index, num in enumerate(w_matrix[0])]),
                    "+".join([str(num) + "λ" + str(index + 1) for index, num in enumerate(w_matrix[1])])]
        print(w_symbol)
        parameter_matrix = []
        # print("the matrix")
        for i in range(len(supporting_vectors)):
            temp = []
            for index in range(len(w_matrix)):
                temp.append(w_matrix[index] * supporting_vectors[i][index])
                # print(w_matrix[index], supporting_vectors[i][index], w_matrix[index] * supporting_vectors[i][index])
            sub_parameter = list(np.sum(temp, axis=0))
            sub_parameter.append(1)
            parameter_matrix.append(sub_parameter)
        # another equation is the one w0 one
        classes.append(0)
        w0_equation = np.array(classes)
        parameter_matrix.append(w0_equation)
        result = np.array(classes)
        unknowns = self.equation_solver(parameter_matrix, result)
        w = []
        for index, vector in enumerate(w_matrix):
            w.append(round(sum([vector[i] * unknowns[i] for i in range(len(vector))]), 4))
        w.insert(0, unknowns[-1])
        print("The parameter of this SVM is {}".format(w))
        print(["w{} = {}".format(index, item) for index, item in enumerate(w)])
        print("**************************************************************** " +
              "End of Function SVM ****************************************************************\n\n")
        return parameter_matrix, unknowns, w

    def mapping_to_another_space(self, data, mapping_function):
        result = []
        for datum in data:
            result.append(mapping_function(datum))
        return result

    def bagging(self, dataset, label, weak_functions, threshold, sgn):
        """
        In practice, dataset will be divided for training, but within this question, the functions are given for sure,
        :param x: the data
        :param weak_functions: Generally, the weak_functions are learned, but within this function. They are given.
        :return:
        """
        key_value_mapping = {function: index for index, function in enumerate(weak_functions)}
        print("******************************** Start of Function Bagging ********************************")
        while True:
            summation_list = []
            temp_list = weak_functions.copy()
            for index_function, weak_function in enumerate(weak_functions):
                function_classification_result = []
                training_error = 0
                for index, x in enumerate(dataset):
                    function_classification_result.append(weak_function(x))
                    if weak_function(x) != label[index]:
                        training_error += 1
                training_error = training_error / len(dataset)
                if training_error > threshold:
                    temp_list.remove(weak_function)
                summation_list.append(function_classification_result)
            summation_list = np.array(summation_list).T
            summation_list = summation_list.sum(axis=1) / 8
            for index in range(len(summation_list)):
                summation_list[index] = sgn(summation_list[index])
            global_training_error = 0
            for index, item in enumerate(summation_list):
                if summation_list[index] != label[index]:
                    global_training_error += 1
            global_training_error /= len(summation_list)
            print("The training error are {}".format(global_training_error))
            if global_training_error == 0:

                print("The functions are: \n{}".format(
                    ["H{}(x)".format(key_value_mapping[function] + 1) for function in weak_functions]))
                return
            else:
                weak_functions = temp_list.copy()

    def adaboost_algorithm(self, dataset, labels, k_max, functions):
        final_hard_classifier = ""
        k = 0
        w_values = [1 / len(dataset) for _ in range(len(dataset))]
        while k < k_max:
            print("***************************************************Round {}**********************"
                  "*****************************".format(k + 1))
            best_classifier = functions[0]
            best_index = 0
            epsilon_k = float("inf")
            k += 1
            for index, weak_function in enumerate(functions):
                training_error = 0
                predicting_result = []
                for index_element, element in enumerate(dataset):
                    predicting_result.append(weak_function(element))
                    if weak_function(element) != labels[index_element]:
                        training_error += w_values[index_element]
                # print("H{}(x) 's training result is \n {}".format(index + 1, predicting_result))
                # print("The true label should be \n {}".format(labels))
                print("The H{}(x) Training Error is {}".format(index + 1, training_error))
                if epsilon_k > training_error:
                    epsilon_k = training_error
                    best_classifier = weak_function
                    best_index = index + 1
            alpha = round(0.5 * math.log((1 - epsilon_k) / epsilon_k), 6)
            # Update the weights
            z_k = 0  # The summation of weights
            for index in range(len(w_values)):
                new_weights = round(
                    w_values[index] * math.exp(-alpha * labels[index] * best_classifier(dataset[index])), 6)
                z_k += new_weights
                w_values[index] = new_weights
            for index in range(len(w_values)):
                w_values[index] = round(w_values[index] / z_k, 6)
            print(w_values)
            final_hard_classifier += str(round(alpha, 4)) + "*H{}(x) + ".format(best_index)
        print("The final hard classifier function is \n{}".format(final_hard_classifier[:-2]))
        print("***************************** The End of ADABOOST ***************************** ")

    def agglomerative_hierarchical_clustering(self, feature_vector_array, k, distance_function):
        vector_index = {str(item): index + 1 for index, item in enumerate(feature_vector_array)}
        while len(feature_vector_array) > k:
            global_minimum_distance = float("inf")
            global_to_be_clustered_1 = global_to_be_clustered_2 = feature_vector_array[0]
            for index_1 in range(len(feature_vector_array)):
                for index_2 in range(index_1 + 1, len(feature_vector_array)):
                    temp_distance = distance_function(feature_vector_array[index_1], feature_vector_array[index_2])
                    if global_minimum_distance > temp_distance:
                        global_minimum_distance = temp_distance
                        global_to_be_clustered_1 = feature_vector_array[index_1]
                        global_to_be_clustered_2 = feature_vector_array[index_2]
            feature_vector_array.remove(global_to_be_clustered_1)
            feature_vector_array.remove(global_to_be_clustered_2)
            feature_vector_array.append(global_to_be_clustered_1 + global_to_be_clustered_2)
        feature_vector_array_index = []
        for cluster in feature_vector_array:
            item_index_list = []
            for item in cluster:
                item_index_list.append(vector_index[str([item])])
            feature_vector_array_index.append(item_index_list)
        print(feature_vector_array_index)

        return feature_vector_array
