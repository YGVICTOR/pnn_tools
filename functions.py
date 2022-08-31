import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import math
from scipy.signal import correlate2d
from numpy import linalg as LA
import sympy as sp
import pandas as pd
# from scipy import linalg


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


class solver():
    def __init__(self):
        pass

    # week 1
    def draw_confusion_matrix(self,predicted_class,true_class):
        print("*****************start performance metrics************************************")
        print()
        result = {"TP": 0,
                  "FN": 0,
                  "FP": 0,
                  "TN": 0
                  }
        for i in range(len(predicted_class)):
            if predicted_class[i]==1 and true_class[i]==1:
                result["TP"]+=1
            elif predicted_class[i] == 0 and true_class[i] ==1:
                result["FN"] +=1
            elif predicted_class[i]==1 and true_class[i]== 0:
                result["FP"] +=1
            else:
                result["TN"]+=1
        print("%20s%20s%20s" % ("predicted labels", "True", "False"))
        print("%20s%20s%20s"%("true labels 1","TP=" +str(result["TP"]),"FN="+str(result["FN"])))
        print("%20s%20s%20s"%("true labels 0","FP="+str(result["FP"]),"TN="+str(result["TN"])))
        error_rate = float(result['FP']+result['FN'])/(result['TP']+result['FN']+result['FP']+result['TN'])
        print("a. the error-rate=(FP+FN)/(TP+FN+FP+TN)=",error_rate)
        accuracy = float(result["TP"]+result["TN"])/(result['TP']+result['FN']+result['FP']+result['TN'])
        print("b. the accuracy=(TP+TN)/(TP+FN+FP+TN)=",accuracy)
        recall = float(result["TP"])/(result["TP"]+result["FN"])
        print("c. the recall=TP/(TP+FN)=",recall)
        precision = float(result["TP"])/(result["TP"]+result["FP"])
        print("d. the precision=TP/(TP+FP)=", precision)
        f1_score = float(2*result["TP"])/(2*result["TP"]+result["FP"]+result["FN"])
        print("e. the f1 score =2*TP/(2*TP+FP+FN)=", f1_score)
        alternative_f1_score = float(2 *recall*precision) / (recall+precision)
        print("f. alternatively:f1-score =2*recall*precision/(recall+precision)=", alternative_f1_score)
        print("")
        print("******************************end performance metrics************************************")
        return result

    # week 2
    # 2.2
    def dichotimizer(self, weight, bias, feature_vectors):
        def compute_distance_to_decision_boundary():
            return
        print("***************************Start the dichotimizer without augmentation***********************")
        print("At decision boundary g(x) = 0, therefore decision boundary is defined by:WX + w0 = 0")
        print("{}x+({})=0".format(weight,bias))
        assert len(weight) == len(feature_vectors[0])
        label_hat_list = []
        for vector in feature_vectors:
            label_hat = bias
            for i in range(len(weight)):
                label_hat += weight[i] * vector[i]
            label_hat_tag = 1 if label_hat > 0 else 2
            label_hat_list.append({str(vector): {"value": label_hat, "label_hat": label_hat_tag,"distance":abs(label_hat)/np.linalg.norm(np.array(weight))}})
        for label in label_hat_list:
            print(label)
        print()
        print("Note, the vector normal to the hyperplane points  towards class 1. "+
            "\nThe value of g(x) provides a measure of how far x is from the decision boundary.\n"+
              " The actual distance is given by |g(x)|/||w||")
        print("***************************End the dichotimizer without augmentation***********************")
        return label_hat_list
    #
    # 2.2 augmented
    def augmented_dichotimiezer(self, a_t, y):
        """

        :param a: it's a, not transpose one.
        :param y:
        :return:
        """
        print("***************************Start the dichotimizer with augmentation***********************")
        print("At decision boundary g(x) = 0, therefore decision boundary is defined by:WX + w0 = 0")
        print("{}[1,x]=0".format(a_t))
        label_hat_list = []
        for vector in y:
            vector = np.transpose(vector)
            label_hat = np.matmul(a_t, vector)
            label_hat_tag = 1 if label_hat > 0 else 2
            label_hat_list.append({str(vector): {"value": label_hat, "label_hat": label_hat_tag,"distance":abs(label_hat)/np.linalg.norm(np.array(a_t)[1:])}})
        for label in label_hat_list:
            print(label)
        print()
        print("Note, the vector normal to the hyperplane points  towards class 1. " +
              "\nThe value of g(x) provides a measure of how far x is from the decision boundary.\n" +
              " The actual distance is given by |g(x)|/||w||")
        print("***************************End the dichotimizer with augmentation***********************")

        return label_hat_list
    #
    # 2.6
    def batch_perceptron_learning_algorithm_with_normalisation(self, x_list, a_t, learning_rate):
        """
        This algorithm is only for linear separably
        :param x_list:
        :param label:
        :param a_t:
        :param learning_rate:
        :return:
        """
        print("***************************Start the Batch Perceptron Learning Algorithm with augmentation and normalization***********************")
        print("For the Batch Perceptron Learning Algorithm, weights are updated such that a <- a + learning_rate * sigma(y)")
        print("y = f(x)，就是训练的parameter 乘的旁边那个数，discriminator g(x) = parameter * y(x)，根据discriminator的正负判定是否misclassification.\n Normalization 之后，g(x)必须大于0才是correctly classified")
        print("update rule : 直接加mismatch 的 y")
        score = 0  # score plus 1 when there is a true match
        epoch = 0
        while score != len(x_list):
            score = 0
            miss_match = []
            epoch += 1
            print("epoch {}".format(epoch))
            print("a_t value before epoch {} = {}".format(epoch,a_t))
            print("%20s%30s%50s" % ("y", "g(x) = a_t * y", "misclassified (g(x) <= 0))"))
            for index, vector_t in enumerate(x_list):
                print("%20s" % vector_t, end="     |      ")
                vector = np.transpose(vector_t)
                label_hat = np.matmul(a_t, vector)
                print("%30s" % "+".join(
                    ["({} * {})".format(vector_t[i], a_t[i]) for i in range(len(vector_t))]),
                      end="")
                print("={}".format(np.matmul(a_t, vector)), end="")
                # since it's been sample normalised, g(x) < 0 is a miss classification
                if not label_hat > 0:  # a miss
                    # if it's a miss
                    miss_match.append(vector)
                    print("%30s" % "yes")
                else:
                    score += 1
                    print("%30s" % "no")
            # in the end, update the a_t value
            print("Updated the a_t value", end=" ")
            a_t = np.transpose(np.transpose(a_t) + learning_rate * sum(miss_match))
            print(a_t)
        print("\nLearning has converged, so required parameters are a = {}".format(a_t))
        print("***************************End the Batch Perceptron Learning Algorithm with augmentation and Normalization ***********************")
        return a_t

    # 2_6
    def batch_perceptron_learning_algorithm_without_sn(self, a_t, x_list, label, learning_rate):
        print(
            "***************************Start the Batch Perceptron Learning Algorithm with augmentation***********************")
        print(
            "For the Batch Perceptron Learning Algorithm, weights are updated such that a <- a + learning_rate * sigma(y)")
        print("y = f(x)，就是训练的parameter 乘的旁边那个数，discriminator g(x) = parameter * y(x)，根据discriminator的正负判定是否misclassification.\n 没有Normalization ，g(x)的正负和true label匹配就是correctly classified")
        print("update rule : 直接加mismatch 的 y")
        score = 0
        epoch = 0
        while score != len(x_list):
            epoch += 1
            print("epoch: {}".format(epoch))
            print("a_t before epoch {}: {}".format(epoch,a_t))
            print("%10s%15s%20s%20s%40s"%("true_labels","y","g(x) = ay","predicted_labels","misclassified(g(x)<=0)"))
            score = 0
            miss_match = []
            for index, vector_t in enumerate(x_list):
                print("%10s"%label[index],end="")
                print("%15s"%(vector_t),end="")
                vector = np.transpose(vector_t)
                label_hat = 1 if np.matmul(a_t, vector) > 0 else -1
                print("%20s" %("+".join(["{}*{}".format(a_t[i],vector_t[i]) for i in range(len(a_t))]))+"={}".format(np.matmul(a_t,vector)),end="")
                # since it's not been sample normalised, label_hat not equal = label is a mismatch
                print("%20s"%(label_hat),end="")
                misclassified = "No" if label_hat == label[index]else"Yes"
                print("%40s"%(misclassified))
                if label_hat != label[index]:  # a miss
                    # if it's a miss
                    miss_match.append(label[index] * np.array(vector))
                else:
                    score += 1
            # in the end, update the a_t value
            a_t = np.transpose(np.transpose(a_t) + learning_rate * sum(miss_match))
        print("\nLearning has converged, so required parameters are a = {}".format(a_t))
        print("***************************End the Batch Perceptron Learning Algorithm without sample normalization***********************")
        return a_t

    # 2_7
    # By default, it's been normalised
    def sequential_perceptron_learning_algorithm_with_sample_normalization(self, a_t, x_list, learning_rate):
        """
        label_hat = a_t * y
        a_t = a_t + learning_rate * yk where yk is wrongly label feature.
        :param a_t: augmented variable a_t = [w_0 wT]_T
        :param y: augmented variable y = [1 xT]_T
        :param learning_rate:
        :return:
        """
        print("***************************Start the Sequential Perceptron Learning Algorithm with Normalization***********************")
        print("For the Sequential Perceptron Learning Algorithm, weights are updated such that:\n a <- a + learning_rate* yk, where yk is a misclassified exemplar.")
        print("since it's been sample normalised, g(x) < 0 is a miss classification")
        score = 0
        epoch = 0
        while score != len(x_list):
            score = 0
            epoch += 1
            print("epoch {}".format(epoch))
            print("before epoch {}, a ={} ".format(epoch,a_t))
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
                    score += 1
        print("Learning has converged, so required parameters are a = {}".format(a_t))
        print("***************************End the Sequential Perceptron Learning Algorithm with Normalization***********************")
        return a_t
    #
    # When it's not sample normalised
    def sequential_perceptron_learning_algorithm_without_sample_normalization(self, a_t, x_list, label, learning_rate):
        """
        label_hat = a_t * y
        a_t = a_t + learning_rate * yk where yk is wrongly label feature.
        :param a_t: augmented variable a_t = [w_0 wT]_T
        :param y: augmented variable y = [1 xT]_T
        :param learning_rate:
        :return:
        """
        print("***************************Start the Sequential Perceptron Learning Algorithm without Sample Normalization***********************")
        print("For the Sequential Perceptron Learning Algorithm, weights are updated such that:\n a <- a + learning_rate* yk, where yk is a misclassified exemplar.")
        print("没有Normalization ，g(x)的正负和true label匹配就是correctly classified")
        score = 0
        itertion = 0
        print("%10s%15s%15s%40s%20s%20s%40s" % ("iteration","a_old","y_k", "g(x) = ay","predicted_labels","true_labels", "update :a <- a + learning_rate* yk"))
        while score != len(x_list):
            score = 0
            for index, vector_t in enumerate(x_list):
                itertion+=1
                print("%10s"%(itertion),end="")
                print("%15s"%(a_t),end="")
                print("%15s"%(vector_t),end="")
                vector = np.transpose(vector_t)
                print("%40s"%(" + ".join([str(a_t[i]) + "*" + str(vector_t[i]) for i in range(len(vector_t))]) + "="+str(np.matmul(a_t, vector))),end="")
                # since it's not been sample normalised, g(x) > 0 indicates it belogs to +1,while g(x)<=0 indicates it belongs to -1
                label_hat = 1 if np.matmul(a_t, vector) > 0 else -1
                print("%20s"%(label_hat),end="")
                print("%20s"%(label[index]),end="")
                if label_hat != label[index]:  # a miss
                    # since it's a sequential gradient descent, the a_t changes immediately
                    # a_t += np.transpose(learning_rate * vector) * label[index]
                    print("%40s" % (str(a_t) + "+" + str(learning_rate) + "*" + str(vector) + "="), end="")
                    a_t += np.transpose(learning_rate * vector)* label[index]
                    print(a_t)
                else:
                    print("%40s" % (a_t))
                    score += 1
        print("Learning has converged, so required parameters are a = {}.".format(a_t))
        print("***************************End the Sequential Perceptron Learning Algorithm without Sample Normalization***********************")
        return a_t

    # 2_11
    def Sequential_Multiclass_Perceptron_Learning_algorithm_without_sample_normalization(self,a_t, x_list, label, learning_rate,class_numbers):
        def return_predict_class(dynamic_g_dict):
            tmp_result = [(k,v) for k,v in dynamic_g_dict.items()]
            tmp_result.sort(key=lambda x: x[1])
            return tmp_result[-1][0]

        print("***************************Start the Sequential Multiclass Perceptron Learning algorithm***********************")

        # revise
        a_t = [[-0.5, 2.0, 1.5],
               [-3.0, -0.5, 0.0],
               [0.5, 0.5, 0.5]]

        dict_a = {}
        for i in range(class_numbers):
            name = "a_{}".format(i+1)
            # revise
            dict_a[name] = a_t[i]
            # dict_a[name] = a_t
        score = 0
        itertion = 0
        print("%10s"%("iteration"),end="")
        for i in range(class_numbers):
            print("%15s"%("a_{}_old".format(i+1)),end="")
        print("%15s"%("y_t"),end="")
        for i in range(class_numbers):
            print("%15s"%("g_{}".format(i+1)),end="")
        print("%15s"%("true_label(ω)"),end="")
        for i in range(class_numbers):
            if i != class_numbers -1:
                print("%15s" % ("a_{}_new".format(i + 1)), end="")
            else:
                print("%15s" % ("a_{}_new".format(i + 1)))
        while(score !=len(x_list)):
            score = 0
            for index, vector_t in enumerate(x_list):
                itertion+=1
                print("%10s"%(itertion),end="")
                for i in range(class_numbers):
                    name = "a_{}".format(i + 1)
                    print("%15s"%(dict_a[name]),end="")
                print("%15s"%(vector_t),end="")
                dynamic_g_dict = {}
                for i in range(class_numbers):
                    name = "a_{}".format(i+1)
                    vector = np.array(vector_t).T
                    dynamic_g = np.matmul(dict_a[name],vector)
                    dynamic_g_dict[i+1] = dynamic_g
                    print("%15s"%(dynamic_g),end="")
                print("%15s"%(label[index]),end="")
                predicted_label = return_predict_class(dynamic_g_dict)
                flag = predicted_label == label[index]
                # 预测错误
                if not flag:
                    a_true_label_name = "a_{}".format(label[index])
                    a_predicted_label_name = "a_{}".format(predicted_label)

                    dict_a[a_true_label_name] = np.array(dict_a[a_true_label_name])
                    dict_a[a_true_label_name] =dict_a[a_true_label_name]+ np.array(vector_t)*learning_rate
                    dict_a[a_true_label_name] = list(dict_a[a_true_label_name])

                    dict_a[a_predicted_label_name] = np.array(dict_a[a_predicted_label_name])
                    dict_a[a_predicted_label_name] =dict_a[a_predicted_label_name]- np.array(vector_t)*learning_rate
                    dict_a[a_predicted_label_name] = list(dict_a[a_predicted_label_name])
                # 预测正确
                else:
                    score +=1
                for i in range(class_numbers):
                    name = "a_{}".format(i + 1)
                    if i != class_numbers-1:
                        print("%15s"%(dict_a[name]),end="")
                    else:
                        print("%15s" % (dict_a[name]))
        print("Learning has converged, so required parameters are")
        for i in range(class_numbers):
            name = "a_{}".format(i+1)
            print("a_{} = {}".format(i+1,dict_a[name]))
        print("***************************End the Sequential Multiclass Perceptron Learning algorithm***********************")

    # 2_12
    def pseudoinverse_with_sample_normalization(self,y,b):
        print("***************************Start the pseudoinverse with sample normalization***********************")
        print("Ya = b where Y =\n",y)
        print("Find the pseudo-inverse of Y' =(Y_transpose Y)' Y_transpose =\n Y'=\n",np.linalg.pinv(np.array(y)))
        print("b = \n",b)
        result = np.matmul(np.linalg.pinv(np.array(y)),np.array(b).T)
        print("Thus, a = Y'b = \n",result.T)
        print('Check')
        print("%10s%40s"%("y_t","           g(x)=ay={}y".format(result)))
        correct =0
        for index, current_y in enumerate(y):
            print("%10s"%(current_y),end="")
            if np.matmul(result,np.array(current_y).T)>0:
                correct +=1
            print("%40s"%(np.matmul(result,np.array(current_y).T)))
        if(correct == len(y)):
            print("All positive, so discriminant function provides correct classification")
        print("***************************End the pseudoinverse with sample normalization***********************")
        return result

    # quiz 2_14
    def sequential_widrow_hoff_learning_with_sample_normalization(self, a_t, margin, feature_vector_t, learning_rate, iteration):
        print("***************************Start the sequential widrow hoff learning ***********************")
        print("For the Sequential Widrow-Hoff Learning Algorithm, weights are updated such that: \n a<- a+η(bk -a_tyk)yk")
        list_a_t = a_t
        a_t = np.array(a_t)
        margin = np.array(margin)
        list_feature_vector = feature_vector_t
        feature_vector_t = np.array(feature_vector_t)
        print("%5s%10s%20s%60s%100s" % ("iteration","a_t","yk", "a_tyk", "at_new = a_t+η(bk - a_tyk)yk"))
        for _ in range(iteration):
            for i in range(len(feature_vector_t)):
                feature_vector = feature_vector_t[i].transpose()
                print("%5s" % (i + 1 + _ * 6), end="")
                print("%15s"%(a_t),end="")
                print("%20s"%(feature_vector_t[i]),end="")
                a_Tyk = np.round((a_t @ feature_vector)[0], 4)
                result = " + ".join(["("+str(a_t[0][j])+" * "+ str(list_feature_vector[i][j])+")" for j in range(len(list_a_t[0]))])
                print("%60s" %(result+"= {}").format(a_Tyk),end="")

                aT_new = np.round(a_t + learning_rate * (margin[i] - a_Tyk) * feature_vector.transpose(), 4)
                # at_new = a_t + η(bk - a_tyk)yk
                print("%100s" % ("{} = {} + {}").format("(" + ",".join(["%.4f" % i for i in aT_new[0]]) + ")",
                                                        "(" + ",".join(["%.4f" % i for i in a_t[0]]) + ")",
                                                        "(" + ",".join(["%.4f" % i for i in (learning_rate * (
                                                                margin[i] - a_Tyk) * feature_vector.transpose())]) + ")"
                                                        ))
                a_t = aT_new
        print("***************************End the sequential widrow-hoff learning ***********************")
        return a_t

    def l2_norm(self, vector_1, vector_2):
        assert len(vector_1) == len(vector_2)
        # TO make this numpy array
        vector_1 = np.array(vector_1)
        vector_2 = np.array(vector_2)
        l2 = np.power(np.sum(np.power((vector_1 - vector_2), 2)), 0.5)
        return l2

    # 2_15
    def knn(self, training_x, true_label,new_feature,k):
        from sklearn.neighbors import KNeighborsClassifier
        from collections import Counter
        print("*************************************Start the KNN ***************************************")
        print("kNN is very sensitive to the scale of the feature dimensions!")
        for new_dot in new_feature:
            data_dict={}
            data_dict['Class'] = true_label
            data_dict['feature_vector'] = training_x
            print("Calculate distance of each sample to x:")
            column_name = "Euclidean distance to({})".format(new_dot)
            data_frame = pd.DataFrame(data_dict)
            data_dict[column_name] = data_frame.apply(lambda x:self.l2_norm(x['feature_vector'],new_dot) ,axis=1)
            data_frame = pd.DataFrame(data_dict)
            print(data_frame)
            sorted_data_frame = data_frame.sort_values(by=[column_name],ascending=True).reset_index()
            nearest_neighbour = list(sorted_data_frame.loc[:k-1,'Class'])
            print("The nearest {} neighbour has class label {}.".format(k,nearest_neighbour))
            final_dict = Counter(nearest_neighbour)
            final_result = [(k,v) for k,v in final_dict.items()]
            final_result.sort(key=lambda x: x[1],reverse=True)
            print("Therefore class new sample as {}: ".format(final_result[0][0]))
            print("\n")
        print("the following results are produced by the KNeighborsClassifier for references.")
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(training_x, true_label)
        for x in new_feature:
            y_hat = neigh.predict(np.array(x).reshape(1, -1))
            print(x, y_hat)
        print("*************************************End the KNN ***************************************")

    # 3_2
    def heaviside_func(self, value, threshold=0):
        if value - threshold> 10e-5:
            return 1
        elif abs(value-threshold)<=10e-5:
            return 0.5
        else:
            return 0

    # Symmetric hard limit
    def symmetric_hard_limit(self,z):
        if z>=10e-5:
            return 1
        else:
            return -1

    # linear transfer
    def linear_transfer(self,z):
        return z


    # Symmetric sigmoid transfer function
    def symmetric_sigmoid(self,z):
        return 2.0/(1+np.exp(-2*z)) - 1

    # Logarithmic sigmoid transfer function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Radial basis transfer function
    def radial_basis(self,z):
        return np.exp(-1*np.square(z))


    def relu(self, net):
        """

        :param net: where net is not a scalar.
        :return:
        """
        net = np.array(net)
        net[net < 10e-5] = 0
        return net

    def tanh(self, net):
        net = np.round(np.tanh(net), 4)
        return net

    def lrelu(self, net, alpha):
        net = np.array(net)
        net[net < 10e-5] = net[net < 10e-5] * alpha
        return net

    def calculate_output_of_a_neuron_with_augmentation_without_sample_normalization(self,w,x,activation_function,**args):
        print("*************************************Start the calculate_output_of_a_neuron_with_augmentation_without_sample_normalization ***************************************")
        print("Output of neuron is defined as:y = H(wx - theta)")
        print("Using {} as activation function H(x)".format(activation_function) )
        for index,current_x in enumerate(x):
            tmp_reslut = np.matmul(np.array(w),np.array(current_x).T)
            display = "H("+ " + ".join(["("+str(w[i])+" * " + str(current_x[i])+")" for i in range(len(current_x))])+")=" +"H({})".format(np.round(tmp_reslut,4)) +" = "
            if activation_function == 'Heaviside':
                result = self.heaviside_func(tmp_reslut,threshold=args['threshold'])
            elif activation_function =='sigmoid':
                result = self.sigmoid(tmp_reslut)
            elif activation_function == 'relu':
                result = self.relu(tmp_reslut)
            elif activation_function =="tanh":
                result = self.tanh(tmp_reslut)
            elif activation_function =="lrelu":
                result = self.lrelu(tmp_reslut,args['alpha'])
            display += str(np.round(result,4))
            print(display)
        print("*************************************End the calculate_output_of_a_neuron_with_augmentation_without_sample_normalization ***************************************")

    def sequential_delta_learning_algorithm_stop_when_all_match(self, w, label, x, learning_rate=1):
        """
        in this algorithm, parameters must be augmented first.
        :param w:
        :param label:
        :param x:
        :return:
        """
        print("*************************************Start the sequential_delta_learning_algorithm_stop_when_all_match ***************************************")
        print("Using Augmented notation, y = H(wx) where w = [-theta,w1], and x = [1; x]T .")
        print("For the Delta rule, weights are updated such that: w <- w + η(t-H(wx))x")
        w = np.array(w)
        x = np.array(x)
        score = 0
        assert len(label) == len(x)
        epoch = 0
        while score != len(label):
            epoch += 1
            print("epoch {}".format(epoch))
            print("w before epoch :{} is {}".format(epoch,w))
            print("%5s%5s%40s%20s%20s%20s" % ("x_t", "t", "y=H(wx)", "t-y", "n(t-y)x_t", "w"))
            score = 0
            for index, vector_t in enumerate(x):
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
                print("%15s" % (str(label[index]) + "-" + str(label_hat)+" = "+str(dif)), end="")
                new_w = w + learning_rate * dif * np.array(vector_t)
                print("%30s" % (str(learning_rate) + "*" + str(dif) + "*" + str(vector_t) + '=' + str(
                    learning_rate * dif * np.array(vector_t))), end="  ")
                print("%5s" % (str(new_w)))
                w = new_w
                if label[index] == label_hat:
                    score += 1
        print("The FINAL w is {}".format(w))
        print(w)
        try:
            display_w = ", ".join(["w_{} = ".format(i)+str(w[0][i]) for i in range(1,len(w[0]))])
            print("Learning has converged, so required weights are w = {}, or equivalently theta = {} , {}  .".format(w,-1*w[0][0],display_w))
        except:
            display_w = ", ".join(["w_{} = ".format(i)+str(w[i]) for i in range(1,len(w))])
            print("Learning has converged, so required weights are w = {}, or equivalently theta = {} , {}  .".format(w,-1*w[0],display_w))
        finally:
            print("*************************************End the sequential_delta_learning_algorithm_stop_when_all_match ***************************************")
        return w

    def row_wise_normalised(self, ori_matrix):
        return normalize(ori_matrix, norm="l1")

    # stop when no changes
    def sequential_delta_learning_algorithm_stop_when_no_changes(self, w, label, x, learning_rate=1):
        """
        in this algorithm, parameters must be augmented first.
        :param w:
        :param label:
        :param x:
        :return:
        """
        w = np.array(w)
        x = np.array(x)
        assert len(label) == len(x)
        while True:
            old_w = w
            for index, vector_t in enumerate(x):
                vector = vector_t.transpose()
                new_w = w + learning_rate * (label[index] - self.heaviside_func(np.matmul(w, vector))) * np.array(
                    vector_t)
                w = new_w
            if np.array_equal(new_w, old_w):
                return w

    def check_all_match(self, w, x, label):
        assert len(label) == len(x)
        for index, vector_t in enumerate(x):
            label_hat = np.matmul(w, np.transpose(vector_t))
            print(label_hat)
            if label[index] != self.heaviside_func(label_hat):
                return False
        return True

    def batch_delta_learning_rule(self, w, x_list, label, learning_rate=1):
        print("*************************************Start the batch_delta_learning_algorithm_stop_when_all_match ***************************************")
        score = 0
        epoch = 0
        while score != len(x_list):
            score = 0
            epoch += 1
            print("epoch {}".format(epoch))
            print("before epoch {}, w {}".format(epoch,w))
            pool = []
            print("%5s%5s%40s%20s%20s%10s" % ("x_t", "t", "y=H(wx)", "t-y", "n(t-y)x_t", "w"))
            for index, vector_t in enumerate(x_list):
                vector = np.transpose(vector_t)
                trans_value = np.matmul(w, vector)
                label_hat = self.heaviside_func(trans_value)
                # since it's not been sample normalised, label_hat not equal = label is a mismatch
                print("%5s" % (vector_t), end="")
                print("%5s" % (label[index]), end="")
                try:
                    print("%30s" % (
                            "H(" + "+".join(["({} * {})".format(vector_t[i], w[0][i]) for i in range(len(vector_t))])),
                          end=")=")
                except:
                    print("%30s" % (
                            "H(" + "+".join(["({} * {})".format(vector_t[i], w[i]) for i in range(len(vector_t))])),
                          end=")=")
                trans_value = np.matmul(w, vector)
                print(("H({})").format(trans_value), end="=")
                label_hat = self.heaviside_func(trans_value)
                print(label_hat, end="")
                dif = label[index] - label_hat
                print("%15s" % (str(label[index]) + "-" + str(label_hat))+"="+str(dif), end="")
                display = list(learning_rate * (label[index] - label_hat) * np.array(vector_t))
                print("%30s" % (str(learning_rate) + '*' + "(" + str(dif) + ")"+"*"+str(vector_t)+"="+str(display)))
                if label_hat != label[index]:
                    pool.append(learning_rate * (label[index] - label_hat) * np.array(vector_t))
                else:
                    pool.append(learning_rate * (label[index] - label_hat) * np.array(vector_t))
                    score += 1
            print("total weight change = %s" % sum(pool))
            # in the end, update the a_t value
            w = w + learning_rate * sum(pool)
            print("after epoch{} , w is {}".format(epoch,w))
            print()

        try:
            display_w = ", ".join(["w_{} = ".format(i)+str(w[0][i]) for i in range(1,len(w[0]))])
            print("Learning has converged, so required weights are w = {}, or equivalently theta = {} , {}  .".format(w,-1*w[0][0],display_w))
        except:
            display_w = ", ".join(["w_{} = ".format(i)+str(w[i]) for i in range(1,len(w))])
            print("Learning has converged, so required weights are w = {}, or equivalently theta = {} , {}  .".format(w,-1*w[0],display_w))
        finally:
            print("*************************************End the batch_delta_learning_algorithm_stop_when_all_match ***************************************")

        return w

        # 3.7, 3.8

    def negative_feedback_neural_network_update_e_only(self, weights, update_rate, x_t, y_t, iteration=5):
        print("*************************************Start the negative_feedback_neural_network_update_e_only  ***************************************")
        print("The activation of a negative feedback network is determined by iteratively evaluating the following equations:")
        print("e = x - W_T * y;  e 维度(input_维度*1);  x维度(input*1) W维度（output * input） y维度（output *1）")
        print("y -> y + alpha *We   We维度（output*1）")
        print("W -> W + βy(eT)   y维度（output*1）eT 维度(1*input_维度);")
        weights = np.round(np.array(weights), 4) # weight = (output * input)
        x_t = np.round(np.array(x_t), 4) # x_t =(1*input)
        weights_t = np.transpose(np.round(np.array(weights), 4)) # weight_t = (input * output)
        x = np.transpose(x_t) # x = (input *1)
        y = np.transpose(y_t) # y = （output*1）
        print("Original y is :{}".format(y.T))
        print("%5s%20s%25s%20s%30s" % ("iteration", "e_T", "(We)T (Y的增量)（+）", "y_T", "(W_Ty)T (e的增量（-）)"))
        for _ in range(iteration):
            print("%5s" % (_ + 1), end="")
            e = np.round(x - weights_t @ y, 4) # e(input *1)
            print("%30s" % (e.transpose()), end="")
            print("%20s" % ((weights @ e).transpose()), end="") # We是y的增量
            y = np.round(y + update_rate * weights @ e, 4) # 更新后的y
            print("%20s" % (y.transpose()), end="")
            print("%30s" % (weights_t @ y).transpose()) # wt y是x的增量
        # print("Finally the Y is \n{}".format(y))
        print("So output is Y =\n{}".format(y))
        print("可以看到(W_Ty)T变得和input x非常接近，(W_Ty)T向着重构input x的方向converge")
        print("根据y_T中元素大小的变化关系识别哪些部分被suppressed，变小的部分就是被suppressed的部分")
        print("If alpha is too large the network becomes unstable. \nInstability is a common problem with recurrent neural networks.")
        print("*************************************End the negative_feedback_neural_network_update_e_only  ***************************************")
        return y

    # 3.9
    def Regulatory_feedback(self, weights, x_t, y_t, sig_1, sig_2, iteration=5):
        print("*************************************Start the Regulatory feedback  ***************************************")
        print( "sig_1 影响y_t的值")
        print("sig_2 影响 e_T的值")
        print("Initial y_t is {}".format(y_t))
        print("%10s%25s%35s%30s%30s" % ("iteration", "e_T", "(W_hat e)T (Y的增量)（+）", "y_T", "(W_Ty)T (e的增量（-）)"))
        for current_iteration in range(iteration):
            print("%10s"%(current_iteration+1),end="")
            Wy = np.matmul(np.array(weights).T, np.array(y_t).T) # (input*output) *(output*1) = (input*1)
            Wy_max_sig2 = [np.round(max(Wy[i],sig_2),6) for i in range(len(Wy))]
            e =[np.round(x_t[i]/Wy_max_sig2[i],6) for i in range(len(Wy_max_sig2))]
            print("%25s"%(e),end="")
            # 归一化W
            W_hat = np.array(weights)/(np.sum(np.array(weights),axis=1).reshape(len(weights),1)) # w_hat =(output*input)
            W_hat_e = np.round(np.matmul(W_hat,e),6)
            print("%35s"%(W_hat_e.T),end="")
            # 计算y
            y_max_sig1 = [max(y_t[i],sig_1) for i in range(len(y_t))]
            y_t = list(np.round(np.multiply(np.array(y_max_sig1),W_hat_e),6))
            print("%35s"%(y_t),end="")
            W_Ty_transpose = np.matmul(np.array(weights).T,np.array(y_t).T).T
            print("%35s"%(W_Ty_transpose))
        print("So output is\n y= {}".format(y_t))
        print("可以看到(W_Ty)T变得和input x非常接近，(W_Ty)T向着重构input x的方向converge")
        print("根据y_T中元素大小的变化关系识别哪些部分被suppressed，变小的部分就是被suppressed的部分")
        print("*************************************End the Regulatory feedback  ***************************************")
        return y_t

    # week4
    def fully_connected_neuron_network_architecture(self,architecture):
        print("*************************************Start the fully_connected_neuron_network_architecture  ***************************************")
        print("the number of units in the input layer is : {} \n The number of units in each corresponds hidden layer is :{} \n The number of units in ouput_layer is : {}".format(architecture[0],architecture[1:len(architecture)-1],architecture[len(architecture)-1]))
        print("\nNow we need to add 1 more bias unit to each layer")
        print("the number of units in the input layer is : {} \n The number of units in each corresponds hidden layer is :{} \n The number of units in ouput_layer is : {}".format(architecture[0]+1,list(np.array(architecture[1:len(architecture)-1])+1),architecture[len(architecture)-1]))
        connections = [architecture[i]*(architecture[i-1]+1)  for i in range(1,len(architecture))]
        print("\narchitecture is :")
        result =  "".join([str(connections[index])+" -> hidden_layer_{}-> ".format(index+1) if index !=len(connections)-1 else str(connections[index])+"-> output_layer" for index in range(len(connections))])
        result = "input_layer->" +result
        print(result)
        print("Total number of weights: ")
        display = " + ".join([str(connections[i]) for i in range(len(connections))]) + " = {}".format(np.sum(np.array(connections)))
        print(display)
        print("*************************************End the fully_connected_neuron_network_architecture  ***************************************")
        return np.sum(np.array(connections))

    # for Tutorial 4 question 4
    def feed_forward_NN(self, input_array_t,W_h_i, W_o_h,b_h_i_t,b_o_h_t,a_input,a_hidden,a_output,verbose=True):
        print("*************************************Start the feed_forward_NN  ***************************************")
        W_hidden_input = np.c_[np.array(W_h_i),np.array(b_h_i_t).T]
        W_output_hidden = np.c_[np.array(W_o_h),np.array(b_o_h_t).T]
        input_with_bias = np.insert(np.array(input_array_t),len(input_array_t[0]),1,axis=1)
        if verbose:
            for index,z in enumerate(input_with_bias):
                print("begin processing : {}".format(z))
                print("input layer")
                print("input x_{} = {} (with bias 1 at the end) into the input unit.".format(index+1,z))
                a_1 = a_input(z)
                print("the output of the input unit is:{} ".format(a_1))
                print("hidden_layer")
                hidden_input = np.matmul(W_hidden_input,a_1)
                print("the input to the {} hidden unit respective is {}".format(len(W_h_i),hidden_input))
                y = np.round(a_hidden(hidden_input),4)
                print("the output of the hidden unit y = {}".format(y))
                print("output layer")
                y_with_bias_t = np.insert(y,y.shape[0],1,axis=0)
                output_layer_input = np.matmul(W_output_hidden,y_with_bias_t.T)
                print("the input to the output layer's {} units respectively is {}.".format(output_layer_input.shape[0],output_layer_input))
                final_output = np.round(a_output(output_layer_input),4)
                print("the final output of the output unit:  z= {}".format(final_output))
                print()

    # tutorial 4 question 6
    def radial_basis_function(self, input_layer, centers, sigma_way, hidden_function, label_t):
        print("****************************************Start Radial Basis Function**************************************")
        # There are 3 layers in total.
        label = np.array(label_t).T
        nh = len(centers)
        if sigma_way == "max":
            p = 0
            for index_i in range(len(centers)):
                for index_j in range(index_i + 1, len(centers)):
                    distance = self.l2_norm(centers[index_i], centers[index_j])
                    if distance > p:
                        p = distance
            print("p_max",p)
            sigma = p / math.sqrt(2 * nh)
            print("sigma",sigma)
        elif sigma_way == "avg":
            total_norm_distance = 0
            cnt = 0
            for index_i in range(len(centers)):
                for index_j in range(index_i + 1, len(centers)):
                    total_norm_distance += self.l2_norm(centers[index_i], centers[index_j])
                    cnt+=1
            sigma = total_norm_distance / cnt
            sigma = 2 * sigma
        # revise
        # sigma = 2*sigma
        # sigma = 0.1

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
            "The final weights of would be \n {} \n where the last item of the weight is the intercept".format(weights))
        z = np.matmul(augmented_x,weights)
        # 这个z 就是predicted value，因为w是根据true label反求的，所以此处的值就是true label
        print("predicted labels are")
        print(z)
        print("****************************************End of  Radial Basis Function**************************************")
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
        print("weights:")
        print(weights)
        label = augmented_x @ weights
        print("output_layer",label)
        flat_label = [1 if item > 0.5 else 0 for sublist in label for item in sublist]

        print(" The raw_labels of the input are \n {}".format(flat_label))
        print(" ************************************************ "
              "End of radial_basis_function_given_weights "
              " ************************************************ \n\n\n")


    # week 5
    # 5_4
    def heaviside_matrix(self, net, threshold):
        print("heaviside matrix")
        net = np.array(net)
        net[net - threshold>10e-5] = 1
        net[abs(net - threshold)<=10e-5] = 0.5
        net[net - threshold<-10e-5] = 0
        return net


    # 5_5
    def cnn_batch_normalisation(self, X, beta, gamma, epsilon):
        # The variance is only among the index
        print(
            "*************************************Start cnn_batch_normalisation  ***************************************")

        print("Batch normalisation modifies the output of an individual neuron, x, to become:")
        print("β +γ(x-E(x))/sqrtvar(x)+epsilon)")
        X = np.array(X)
        mean_X = np.mean(X,axis=0)
        var_X = np.var(X,axis=0)

        for i in range(X.shape[0]):
            X[i]= beta +gamma *( (X[i] - mean_X))/np.sqrt(var_X+epsilon)
            print('BN(X_{}) = '.format(i+1))
            print(X[i])

        print( "*************************************End cnn_batch_normalisation  ***************************************")
        return X

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
        print("The Output of the Convolution would be \n {}\n"
              " ************************************************************* ".format(result))
        return result


    # def performance_matrix(self, parameters, feature_vectors, labels):
    #     parameters = np.array(parameters)
    #     feature_vectors = np.array(feature_vectors)
    #     confusion_matrix = {"true positive": 0, "false positive": 0, "true negative": 0, "false negative": 0}
    #     for index, vector_t in enumerate(feature_vectors):
    #         vector = vector_t.transpose()
    #         label_hat = parameters @ vector
    #         label_hat = self.heaviside_func(label_hat)
    #         if label_hat != labels[index]:
    #             if labels[index] == 1:
    #                 confusion_matrix["false negative"] += 1
    #             else:
    #                 confusion_matrix["false positive"] += 1
    #         else:
    #             if labels[index] == 1:
    #                 confusion_matrix["true positive"] += 1
    #             else:
    #                 confusion_matrix["true negative"] += 1
    #     print("Precession {}".format(confusion_matrix["true positive"] / (
    #             confusion_matrix["true positive"] + confusion_matrix["false positive"])))
    #     print("Recall {}".format(confusion_matrix["true positive"] / (
    #             confusion_matrix["true positive"] + confusion_matrix["false negative"])))
    #     return confusion_matrix
    #

    # week 6
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
            expectation_z_pz += math.log((1 - discriminator_function(fake_dateset[index]))) * (
                        1 / len(real_dataset))
        print(expectation_x_pdatax, expectation_z_pz)
        print("Ex~pdata(x)[logD(x)] = {}\n "
              "Ez~pz(z)[ln(1 - D(G(z)))] = {}".format(expectation_x_pdatax, expectation_z_pz))
        print("V(D ,G) = {}".format(expectation_x_pdatax + expectation_z_pz))
        print("****************************************************************"
              "gan_generator_and_discriminator"
              "****************************************************************\n\n\n")

    # week 7
    def oja_rules(self, w, x_features, learning_rate, epoch):
        # w 是unit length,用神经网络降维
        # Using zero-mean data as x.
        # A linear neuron (i.e. a linear threshold unit without the heaviside function)
        # Output, y, is projected data
        # weight decay term causes w to approach unit length

        print("************************Start of the  oja_rules  ***************************")
        # in Oja's rule, and Hebbian rule, they both use the zero-mean to calculate the weights.
        print("initial w at the very begining is: {}".format(w))
        x_features = np.array(x_features)
        mean_vector = np.mean(x_features,axis=0)
        sum_vector = np.sum(x_features,axis=0)
        zero_mean_data = x_features - mean_vector
        w = np.array(w)
        for i in range(epoch):
            delta_w = 0
            print("begin epoch: {}".format(i + 1))
            print("%5s%10s%10s%20s%10s" % ("X_t   ", "y = wx", "x_t-yw", "ny(x_t-yw)", "w"))
            for x in zero_mean_data:
                print('%5s' % (x), end="")
                y = np.sum(np.multiply(w, x))
                print("%5s" % (y), end="")
                print("%10s" % (x - y * w), end="")
                print("%20s" % (learning_rate * y * (x - y * w)))

                delta_w += learning_rate * y * (x - y * w)
            print("\nTotal weight change {}".format(delta_w))
            w = w + delta_w
            print("after epoch {}, initial w = {}".format(i+1, w))
            print('\n')
        print("************************End of the  oja_rules  ***************************")
        return w


    def oja_rules_sequential(self, w, x_features, learning_rate, epoch,decimal=2):
        # w 是unit length,用神经网络降维
        # Using zero-mean data as x.
        # A linear neuron (i.e. a linear threshold unit without the heaviside function)
        # Output, y, is projected data
        # weight decay term causes w to approach unit length

        print("************************Start of the  sequential_oja_rules  ***************************")
        # in Oja's rule, and Hebbian rule, they both use the zero-mean to calculate the weights.
        print("initial w at the very begining is: {}".format(w))
        x_features = np.array(x_features)
        mean_vector = x_features.mean(axis=0)
        zero_mean_data = x_features - mean_vector
        w = np.array(w)
        for i in range(epoch):
            # delta_w = 0
            print("begin epoch: {}".format(i + 1))
            print("%10s%20s%15s%20s%15s" % ("X_t   ", "y = wx", "x_t-yw", "ny(x_t-yw)", "w"))
            for x in zero_mean_data:
                print('%10s' % (x), end="")
                y = np.sum(np.multiply(w, x))
                y = np.round(y,decimals=decimal+2)
                print("%20s" % (y), end="")
                tmp = np.round(x - y * w,decimals=decimal+2)
                print("%20s" % (tmp), end="")
                tmp2 = np.round(learning_rate * y * (x - y * w),decimals=decimal+2)
                print("%20s" % (tmp2),end="")
                delta_w = learning_rate * y * (x - y * w)
                w = w + delta_w
                print("%20s" % (w))
        print("************************End of the  sequential_oja_rules  ***************************")
        return w

    def hebbian_learning_rules_sequential(self,w,x_features,learning_rate,epoch,decimal=2):
        # 用神经网络降维，没有归一，绝对值容易失控
        print("************************Start of the  sequential_hebbian_learning_rules  ***************************")
        print("initial w at the very begining is: {}".format(w))
        x_features = np.array(x_features)
        mean_vector = np.mean(x_features,axis=0)
        zero_mean_data = x_features - mean_vector
        w = np.array(w)
        for i in range(epoch):
            delta_w = 0
            print("begin epoch {}".format(i+1))
            print("%5s%30s%20s%40s" % ("X_t   ", "y = wx",  "ny(x_t)", "w"))
            for x in zero_mean_data:
                print("%5s"%(x),end="")
                y = np.round(np.sum(np.multiply(w, x)),decimals=decimal+2)
                print("%20s"%(y),end="")
                delta_w = learning_rate * y * x
                print("%40s"%(delta_w),end="")
                w = w + delta_w
                w = np.round(w,decimals=decimal+2)
                print("%40s" % w)
        print("************************end of the  sequential_hebbian_learning_rules  ***************************")
        return w


    def kl_transform(self, x_features, principal_numbers: int):
        # kl是传统的PCA,先找特征值，之后根据特征值大小降维
        print("************************Start of the  kl transform ***************************")
        mean_vector = [np.sum(x_features, axis=0) / len(x_features)]
        mean_vector = np.array(mean_vector).transpose()
        print("the mean of the data is： μ=\n", mean_vector)
        intermediate_result = np.array(x_features).T -mean_vector
        print("hence, the zero-mean data is:\n",intermediate_result)
        # a covariance matrix
        C = 1.0/(len(x_features))*np.dot(np.array(x_features).T - mean_vector,intermediate_result.T)
        print("Covariance matrix\n", C)
        # e_values, e_vectors = linalg.eig(np.array(C))
        e_values, e_vectors = LA.eig(np.array(C))
        # Order eigenvalues from large to small, and discard small eigenvalues and their respective vectors
        sorted_e_values, sorted_e_vectors = zip(*sorted(zip(e_values, e_vectors), reverse=True))
        print("Eigenvalue are:\n")
        sorted_e_values = list(sorted_e_values)
        print(np.diag(sorted_e_values))
        print("Eigenvectors are:\n")
        print(np.array(sorted_e_vectors))
        print("Choose the two eigenvectors corresponding to the two largest eigenvectors:\n")
        e_vectors_hat = sorted_e_vectors[:principal_numbers]

        # REVISE
        # 注意，参数是横着传递的；
        # e_vectors_hat =np.array([[0.5777,   0.1085],
        #                 [0.1228,    0.2499],
        #                 [-0.0372,    0.9595],
        #                 [0.8061,   -0.0716]]).T
        e_vectors_hat = np.array([[ -0.7071, -0.7071],
                                  [-0.7071, 0.7071],
                                  [0, 0]
                                  ]).T
        print("e_vectors_hat \n")
        print(np.array(e_vectors_hat).T)
        print("Projection of the data onto the subspace spanned by the first {} principal components\n".format(principal_numbers))
        y_all = (np.array(e_vectors_hat).T).T @ intermediate_result
        print(y_all)
        for index,x_t in enumerate(intermediate_result.T):
            y = np.array(e_vectors_hat) @ x_t
            print("y_{}={}@ {}={}".format(index+1,np.array(e_vectors_hat),x_t.T,y.T))
        print("Therefore, after pca, the original data was transformed into:\n")
        print(y_all)

        print("Proportion of the variance is given by sum of eigenvalues for selected components divided by the sum of all eigenvalues.")
        proportion = np.sum(sorted_e_values[:principal_numbers])/np.sum(sorted_e_values)
        print("numerator :" + " +".join(str(i) for i in sorted_e_values[:principal_numbers])+" = "+str(np.sum(sorted_e_values[:principal_numbers])))
        print("denominator: "+" + ".join(str(i) for i in list(sorted_e_values))+" = " +str(np.sum(sorted_e_values)))
        print("proportion: {}".format(proportion))
        print("**************************End of the  kl transform ****************************")
        return y_all


    def kl_transform_given_values(self, x_features, eigenvalues, eigenvectors_t, principal_numbers: int):
        # find the top "principal_numbers" large eigenvalues corresponding eigenvectors.
        eigenvalues = np.array(eigenvalues)
        eigenvectors_t = np.array(eigenvectors_t)
        x_features = np.array(x_features)
        eigenvalues_index = (-eigenvalues).argsort()
        sorted_eigenvectors_t = eigenvectors_t[eigenvalues_index]
        # get the top "principal_numbers" large eigenvalues corresponding eigenvectors.
        reduced_eigenvectors = sorted_eigenvectors_t[:principal_numbers]
        print("The reduced eigenvectors ")
        print(reduced_eigenvectors)
        mean_vector = np.mean(x_features, axis=0)  # Vertical mean
        print("The reduced vectors are as the followings.")
        for index in range(len(x_features)):
            print(reduced_eigenvectors @ (x_features[index] - mean_vector))
        print(" ************************************************ "
              "KLT"
              " ************************************************ ")


    def fisher_method(self, feature_with_class, w):
        # fisher_method 评价LDA效果的；
        print("************************Start of the  fisher method ***************************")
        w = np.array(w)
        class_1 = np.array(feature_with_class[0])
        class_2 = np.array(feature_with_class[1])
        mean_1 = np.mean(class_1,axis=0)
        mean_2 = np.mean(class_2,axis=0)
        print("sample mean for class 1 is :")
        print("( " + " + ".join([str(x) for x in class_1]) + ") / {} = {}".format(class_1.shape[0], mean_1))
        print("sample mean for class 2 is :")
        print("( " + " + ".join([str(x) for x in class_2]) + ") / {} = {}".format(class_2.shape[0], mean_2))

        sb = np.square(np.sum(np.multiply(w,mean_1-mean_2)))
        print("Between class scatter (sb) is :",sb)
        sw = np.sum(np.square(np.sum(np.multiply(w,class_1-mean_1),axis=1)))+np.sum(np.square(np.sum(np.multiply(w,class_2-mean_2),axis=1)))
        print("Within class scatter (sw) is: ",sw)
        costJ = sb/sw
        print("Cost J(w) = sb/sw = ",costJ)
        print("projection of the data into the new feature space defined by the projection weights is:")
        print("%10s%10s%15s" % ("Class   ", "Feaature Vector", "Y=WX"))
        for i in range(len(feature_with_class)):
            for current_instance in np.array(feature_with_class[i]):
                print("%5s"%(i+1),end="")
                print("%10s"%(current_instance),end="")
                info = "{}* {} = {}".format(w,current_instance,np.sum(np.multiply(w,current_instance)))
                print("%30s"%(info))
        print("************************End of the  fisher method ***************************")
        return


    def extreme_learning_machine(self, random_matrix, x_features, output_neuron):
        # 和SVM很像，低维度不可分通过非线性变换转到另一个维度，在这个维度可分；
        print("************************Start of the  extreme learning machine ***************************")
        random_matrix = np.array(random_matrix)
        x_features = np.array(x_features)
        x_features = x_features.transpose()
        # 在最前面插入一行1
        print("If we place all the augmented input patterns into a matrix we have the following dataset:")
        x_features = np.insert(x_features, 0, 1, axis=0)
        print(x_features)
        VX = random_matrix @ x_features
        print("input to the hidden layer is computed by: VX=")
        print("VX\n", VX)
        print("For result above, each column is corresponded to one instance, the number of elements in the column\n corresponds to the input received by the hidden neuron")
        print("\n ")
        print("The response of each hidden neuron to a single exemplar is defined as y = H(vx), where H is the heaviside\n"+
                "function. The response of all {} hidden neurons to all  {} input patterns, is given by:".format(VX.shape[0],VX.shape[1]))
        Y = self.heaviside_matrix(VX, 0)
        print("Y\n", Y)
        Y = np.insert(Y, 0, 1, axis=0)
        print("The response of the output neuron to a single exemplar is defined as z = wy. The response of the output\n"+
            "neuron to all {} input patterns, is given by: Z = wY ={}*\n{}".format(VX.shape[1],output_neuron,Y))
        Z = output_neuron @ Y
        print("Z\n", Z)
        print("************************End of the  extreme learning machine ***************************")
        return Z

    def sparse_coding(self, original_x, y_t, v_t):
        # data is projected into a high-dimensional space in the
        # hope that this will make classes more separable.
        # only a few of the dimensions are used to represent any
        # one exemplar.
        # 给定一个V_t dictionary，再给出一个原来的特征x，评价给出的稀疏code对原来特征x的表示程度；越小越好
        from numpy.linalg import norm
        from math import sqrt
        print("************************Start of the  sparse code encoding ***************************")
        print("To evaluate the sparse code encoding, we need to compute the reconstruction error:||x-VY||2")
        v_t = np.array(v_t)
        y_t = np.array(y_t)
        y = y_t.transpose()
        x = v_t @ y
        print("Vt \n",v_t)
        print("y\n",y)
        print("Vt@y1 \n", x)
        error = norm(original_x - x.T)
        print("sqrt(x-vy) = np.sqrt(np.sum(np.square({} - {})))={}".format(original_x,x.T,error))
        print("||{} - {} ||2 = sqrt({} + {}) = {}".format(original_x, x.T, (original_x - x.T), (original_x - x.T),
                                                          error))
        # smaller the error, better the performance!
        print("************************End of the  sparse code encoding ***************************")

        return error

    def compute_euclidean_diff(self, array1, array2):
        """
        """
        return np.sqrt(np.sum(np.square(np.absolute(array1 - array2))))

    # week 8
    def svm(self,X,Y):
        print("************************Start of the  svm ***************************")
        from sklearn import svm
        clf = svm.SVC(kernel='linear')
        clf.fit(X, Y)
        support_vectors = clf.support_vectors_
        print("support vectors: \n",support_vectors)
        parameter = clf.coef_
        print("parameters are ",parameter)
        margin = 2.0/ np.linalg.norm(parameter)
        print("margin  = 2/||w|| = :",margin)
        interceptor = clf.intercept_
        print("interceptor is ",interceptor)
        print("************************End of the  svm ***************************")
        return clf

    def equation_solver(self, parameters, results):
        inverse_parameters = np.linalg.pinv(np.array(parameters))
        results = np.array(results)
        unknowns = inverse_parameters @ results.T
        return unknowns

    def svm_(self, supporting_vectors, classes):
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
        print("待求的W用λ表示：")
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
        print("非supporting vector lambda是0")
        print("equation about lambda: yi(W_Tz + w0) = 1:")
        print("前几个最后一项是w0，最后一个是lambda之间的关系")
        for i in parameter_matrix:
            print(i)
        print("上面方程等号右侧是：")
        print(result)
        print("lambda和w0解方程的结果是")
        print(unknowns)
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


    # week 9
    def adaboost_algorithm(self, dataset, labels, k_max, functions):
        # 原理是每一个weak classifier 错误划分的点是不会变的，但可以调整权值赋予他们被错误划分的不同代价
        final_hard_classifier = ""
        final_parameter =[]
        classifier = []
        k = 0
        w_values = [1 / len(dataset) for _ in range(len(dataset))]
        print("initial w_values: {}".format(w_values))
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
                # 挑选最小的training error
                if epsilon_k > training_error:
                    epsilon_k = training_error
                    best_classifier = weak_function
                    best_index = index + 1
            if epsilon_k >0.5:
                print("minimum epsilon is bigger than 0.5 , this function sucks")
                k = k_max
                continue
            alpha = round(0.5 * math.log((1 - epsilon_k) / epsilon_k), 6)
            print("epsilion is (min training error)",epsilon_k)
            print("alpha = 1/2ln((1-epsilion)/epsilion )= ",alpha)
            # Update the weights
            z_k = 0  # The summation of weights
            for index in range(len(w_values)):
                new_weights = round(
                    w_values[index] * math.exp(-alpha * labels[index] * best_classifier(dataset[index])), 6)
                z_k += new_weights
                w_values[index] = new_weights
            for index in range(len(w_values)):
                w_values[index] = round(w_values[index] / z_k, 6)
            print("w_values after round {} :\n {}".format(k,w_values))
            final_hard_classifier += str(round(alpha, 4)) + "*H{}(x) + ".format(best_index)
            final_parameter.append(np.round(alpha,4))
            classifier.append(best_classifier)
        print("The final hard classifier function is \n sgn({})".format(final_hard_classifier[:-2]))
        print("***************************** The End of ADABOOST ***************************** ")

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
            # 取一个函数
            for index_function, weak_function in enumerate(weak_functions):
                function_classification_result = []
                training_error = 0
                # 遍历所有的记录
                for index, x in enumerate(dataset):
                    function_classification_result.append(weak_function(x))
                    if weak_function(x) != label[index]:
                        training_error += 1
                training_error = training_error / len(dataset)
                if training_error > threshold:
                    # tmp_list 存合格的weak_classifier
                    temp_list.remove(weak_function)
                # summation_list存一个函数对所有样本的预测结果
                summation_list.append(function_classification_result)
            summation_list = np.array(summation_list).T
            summation_list = summation_list.sum(axis=1) / summation_list.shape[1]
            # 将hi(x)的均值转换为预测tag
            for index in range(len(summation_list)):
                summation_list[index] = sgn(summation_list[index])
            global_training_error = 0
            for index, item in enumerate(summation_list):
                if summation_list[index] != label[index]:
                    global_training_error += 1
            global_training_error /= len(summation_list)
            print("The training error are {}".format(global_training_error))
            # revise
            if global_training_error == 0:

                print("The functions are: \n{}".format(
                    ["H{}(x)".format(key_value_mapping[function] + 1) for function in weak_functions]))
                return
            else:
                weak_functions = temp_list.copy()

    # week 10
    def naive_agglomerative_hierarchical_clustering(self, feature_vector_array, k=3, method='SAD', cluster_method='centroid'):
        """
        """

        feature_vector_array = np.array(feature_vector_array)
        cluster_array = feature_vector_array.reshape(-1, feature_vector_array.shape[-1])
        old_cluster_array = cluster_array.copy()

        record_list = []
        for i in range(1, cluster_array.shape[0] + 1):
            record_list.append([i])
        step = 1
        while cluster_array.shape[0] > k:
            result_list = []
            for i in range(cluster_array.shape[0]):
                cluster_1 = cluster_array[i]
                for j in range(i + 1, cluster_array.shape[0]):
                    cluster_2 = cluster_array[j]
                    dist = self.compute_euclidean_diff(cluster_1, cluster_2)
                    result_list.append([i + 1, j + 1, dist])
            min_dist = min([i[-1] for i in result_list])
            pair_result = []
            for result in result_list:
                if result[-1] <= min_dist:
                    min_dist = result[-1]
                    pair_result.append(result[:-1])
            cluster_result = list()
            symbol = True
            for i in range(1, cluster_array.shape[0] + 1):
                tmp = []
                for pair in pair_result:
                    if i in pair:
                        for val in pair:
                            tmp.append(val)
                for result in cluster_result:
                    if set(result) >= set(tmp):
                        symbol = False
                if symbol:
                    cluster_result.append(list(set(tmp)))
                symbol = True

            new_cluster_index = list()
            for result in cluster_result:
                for i in result:
                    new_cluster_index.append(i)
            new_cluster_array = list()
            new_record_list = list()
            for i in range(len(cluster_result)):
                if not cluster_result[i]:
                    continue
                tmp = []
                for result in cluster_result[i]:
                    target = record_list[result - 1]
                    for m in target:
                        tmp.append(m)
                new_record_list.append(tmp)
                index = np.array(cluster_result[i]) - 1
            for i in range(1, cluster_array.shape[0] + 1):
                if i in new_cluster_index:
                    continue
                else:
                    new_record_list.append(record_list[i - 1])
            for i in range(len(new_record_list)):
                cluster_index_list = np.array(new_record_list[i]) - 1
                if cluster_method == 'centroid':
                    new_cluster_array.append(np.mean(old_cluster_array[cluster_index_list], axis=0))
                else:
                    raise AttributeError("THIS METHOD IS NOR IMPLEMENTED!!!")
            cluster_array = np.array(new_cluster_array)
            record_list = new_record_list
            print("the feature of cluster of {}-th step is\n {}\n".format(step, cluster_array))
            print("the result of cluster of {}-th step is\n {}\n".format(step, record_list))
            step += 1

        print("the result of agglomerative hierarchical clustering is\n {}\n".format(record_list))
        return record_list

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

    def kmeans(self, data_set, initial_center):
        print("************************Start of the  kmeans ***************************")
        ite = 0
        print("initial_center",initial_center)
        while True:
            ite += 1
            result = [[] for i in range(len(initial_center))]
            for data in data_set:
                min_distance = float("inf")
                min_index = -1
                for index, center in enumerate(initial_center):
                    dist = np.linalg.norm(np.array(data) - np.array(center))
                    if dist <= min_distance:
                        min_distance = dist
                        min_index = index
                result[min_index].append(data)
            new_center_tmp = [np.mean(result[i],axis=0) for i in range(len(result))]
            new_center =[]
            for current_center in new_center_tmp:
                new_center.append(list(current_center))
            if np.all(new_center == initial_center):
                break
            initial_center = new_center
            print(initial_center,ite)
            if ite==2:
                print("ite 2")
                print(initial_center)
                print("ite2 finished")
                return


        print("the final k means results is ")
        for i in range(len(result)):
            print("class_{}".format(i+1),result[i])
            print("class_{} centroid: {}".format(i+1,initial_center[i]))
        print("************************End of the  kmeans ***************************")
        return result, initial_center

    def competitive_learning_without_normalization(self,initial_center,learning_rate,iteration,sample_with_order,compute_distance):
        print("************************Start of the  competitive_learning_without_normalization ***************************")
        print("initial centers is \n {}:".format(initial_center))
        initial_center = np.array(initial_center)
        for current_iteration in range(iteration):
            choosen_sample = sample_with_order[current_iteration]
            best_index = 0
            min_distance = float("inf")
            for index, current_center in enumerate(initial_center):
                dist = compute_distance(choosen_sample,current_center)
                if dist < min_distance:
                    best_index = index
                    min_distance = dist
            delta_w = learning_rate *(np.array(choosen_sample) - np.array(initial_center[best_index]))
            initial_center[best_index] = initial_center[best_index] + delta_w
            print("after {} itereation, we have m_{} = {}".format(current_iteration+1,best_index+1,initial_center[best_index]))
        print("\n after {} iteration, we have final cluster centers as \n{}".format(iteration,initial_center))
        print("************************End of the  competitive_learning_without_normalization ***************************")
        return initial_center

    def competitive_learning_with_normalization(self,initial_center,learning_rate,iteration,sample_with_order,compute_distance):
        print("************************Start of the  competitive_learning_with_normalization ***************************")
        print("initial centers is \n {}:".format(initial_center))
        sample_with_order = np.array(sample_with_order)
        normalized_sample_with_order = []
        for index,curr in enumerate(sample_with_order):
            normalized_sample_with_order.append(list(sample_with_order[index]/np.linalg.norm(curr)))
        initial_center = np.array(initial_center)
        for current_iteration in range(iteration):
            choosen_sample = normalized_sample_with_order[current_iteration]
            best_index = 0
            max_distance = -float("inf")
            for index, current_center in enumerate(initial_center):
                dist = np.matmul(current_center,np.array(choosen_sample).T)
                if dist > max_distance:
                    best_index = index
                    max_distance = dist
            delta_w = learning_rate * np.array(choosen_sample)
            initial_center[best_index] = initial_center[best_index] + delta_w
            initial_center[best_index] = initial_center[best_index]/np.linalg.norm(initial_center[best_index])

            print("after {} itereation, we have m_{} = {}".format(current_iteration+1,best_index+1,initial_center[best_index]))
        print("\n after {} iteration, we have final cluster centers as \n{}".format(iteration,initial_center))
        print("************************End of the  competitive_learning_with_normalization ***************************")
        return initial_center

    def given_cluster_center_compute_its_belongs(self,cluster_center,samples,compute_distance):
        result = []
        for current_sample in samples:
            best_index = 0
            min_distance = float("inf")
            for index,current_center in enumerate(cluster_center):
                dist = compute_distance(current_sample,current_center)
                if dist < min_distance:
                    best_index = index
                    min_distance = dist
            result.append([current_sample,best_index+1,min_distance])
        for i in result:
            print("sample:  {} belogs to cluster {}".format(i[0],i[1]))
        return result

    def basic_leader_follower_clustering_without_normalisation(self, dataset, theta, n, pickup_order):
        print("注意：basic leader的第一个cluster center是created出来的,当distance小于threshold 更新当前center\n 当distance大于threshold，创建新的center")
        dataset = np.array(dataset)
        centers = [dataset[pickup_order[0]]]
        print("%10s %20s %50s"%("Sample","Cluster_Center","Distance_between_sample_and_cluster_center"))
        for index in pickup_order:
            print("%10s" % dataset[index], end="")
            print("%20s" % centers, end="")
            flag = True
            min_index = 0
            min_norm2 = float("inf")
            for center_index, center in enumerate(centers):
                temp = np.linalg.norm(dataset[index] - center)
                print("%50s" % "||x-m{}|| = {}".format(center_index + 1, temp), end="     ")
                if min_norm2 > temp:
                    min_index = center_index
                    min_norm2 = temp
            print("%20s" % min_index, end="    ")
            if min_norm2 >= theta:  # add node into center
                centers.append(dataset[index])
                flag = False
            # Only update the chosen center
            if flag:
                centers[min_index] = n * (dataset[index] - centers[min_index]) + centers[min_index]
            print(centers)
        print(centers)
        return centers

    def fuzzy_kmeans(self, dataset, membership, k, b):
        # normalise the membership
        dataset = np.array(dataset)
        centers = []
        for index in range(len(membership)):
            for sub_index in range(len(membership[index])):
                membership[index][sub_index] /= (sum(membership[index]))
        iteration = 1
        while True:
            print(" μ (dimension[1] is cluster number)）")
            print(membership)
            old_center = centers.copy()
            centers = []
            # update the centers
            for i in range(k):
                m_i = np.sum([(item[i] ** b) * dataset[index] for index, item in enumerate(membership)], axis=0) / sum(
                    [item[i] ** b for item in membership])
                centers.append(np.array(m_i))
            print("In iteration {} centers: {}".format(iteration,centers))
            # Update the membership
            for index, item in enumerate(dataset):
                denominator = (sum([(1 / np.linalg.norm(dataset[index] - item)) ** (2/(b-1)) for item in centers]))
                for sub_index, center in enumerate(centers):
                    membership[index][sub_index] = ((1 / np.linalg.norm(dataset[index] - center)) ** (2/(b-1))) / denominator
            iteration+=1
            if len(old_center) == 0:
                continue
            else:

                # print(np.array(centers) - np.array(old_center))
                za = (np.array(centers) - np.array(old_center) > 0.5).sum()
                # print(za)
                if za == 0:
                    break

        print("final centers: ",np.round(centers, 4))