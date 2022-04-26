"""
This program aims to replicate experiment VI conducted in https://arxiv.org/pdf/1901.04609.pdf.

Created by Kevin Pettersson and Reza Qorbani 
16/3-22
"""

import xlsxwriter
import tikzplotlib
import BI_KSG_estimator
import matplotlib.pyplot as plt

from numpy import append, array, mean, ravel, sqrt
from sklearn.linear_model import LogisticRegression
from numpy.random import multivariate_normal, randint


def create_dataset(mean_1, mean_neg_1, covariance, number_of_samples):
    """
    This functions picks out random samples from the distrubution, given 
    the means and covariance. As in our specific case we have 2 distrubutions
    we combine them into 1 array to represent the input and labels.

    Input:
    mean_1 = the mean of the distrubution for the value 1, array of dimension 
    1 or greater 
    mean_neg_1 = the mean of the distrubution for the value -1, array of dimension 
    1 or greater 
    covariance = the covariance for the distrubutions, array -> [[dim mean_1], [dim mean_neg_1]]
    number_of_samples = how many samples should be picked_out

    Output:
    Z = data from distrubution with correpsonding label in 2D array -> [[data1, data1, label1], [data2, data2, label2],....] ex dim = 2
    """

    # checks to make sure the correct data is sent
    if(len(mean_1) != len(mean_neg_1)):
        raise ValueError("Dimensions of means must match!")
    
    if(len(covariance[0]) != len(mean_1) or len(covariance[1]) != len(mean_neg_1)):
        raise ValueError("Covariance must have the same dimensions as the means!")

    #if(number_of_samples % 2 != 0):
    #   raise ValueError("Number of samples must be even!")
    
    # draw random samples form multivariate normal distrubution
    X_1 = multivariate_normal(mean_1, covariance, int(number_of_samples)) 
    X_neg_1 = multivariate_normal(mean_neg_1, covariance, int(number_of_samples))
    X = append(X_1, X_neg_1, 0)
    
    Z_list = X.tolist()
    label = 0
    for i in range(len(X)):
        if(i+1 > int(number_of_samples)):
            label = 1
        Z_list[i].append(label)
    Z = array(Z_list)

    return Z


def train_W(Z):
    """
    Models the probability of the two discrete outcomes given input samples. 
    
    Input:
    Z = input dataset consitsing of samples and labels in 2D array.

    Output:
    W_model = a logistic regression model fitted to the input samples 
    """

    # data handling to seperate input data from labels
    dim = len(Z[0])-1
    X = Z[:,0:dim]
    Y = Z[:,dim:dim+1]
    Y = ravel(Y)

    W_model = LogisticRegression(random_state=None, solver='lbfgs', fit_intercept=False).fit(X, Y) 
    
    return W_model


def test_accuracy(predicted_output, label):
    """
    Tests the accuracy of the predictions made to the 
    test samples. Returns the total average accuracy
    for the predicted outputs.

    Input:
    predicted_output = array with labels of the predicted output
    label = array with the correct output
    
    Note the arrays must be synced so that the index in
    prediced_output matches the same index in label
    """
    
    total = len(predicted_output)
    correct = 0
    for i in range(len(predicted_output)):
        if(predicted_output[i] == label[i][0]):
            correct += 1

    return correct/total


def estimate_gen_error(W_model, Z_test, Z):
    """
    Estimates the generilization error numerically by
    finding the excpected value on the accuracy of the training
    data and subtracting it by the excpected value on the accuracy 
    of the test data.    

    Input:
    W_model = the logistic regression model that is trained/fitted
    Z_test = test dataset 2D array 
    Z = training dataset 2D array

    Output:
    est_gen_error = the numerically estimated generilizaztion error
    """

    # data handling 
    dim = len(Z[0])-1
    X_test = Z_test[:,0:dim]
    Y_test = Z_test[:,dim:dim+1]
    X = Z[:,0:dim]
    Y = Z[:,dim:dim+1]

    # find the models predictions on the datasets
    predicted_output_test = (W_model.predict(X_test))
    predicted_output = (W_model.predict(X))

    # calculate the average accuracy on the datasets
    accuracy_test = test_accuracy(predicted_output_test, Y_test)
    accuracy = test_accuracy(predicted_output, Y)
    
    est_gen_error = accuracy-accuracy_test

    return est_gen_error


def plot_results(x_list, y_list, tkitz = True):
    """
    Graphs the data generated from the experiment.

    Input:
    x_list = x axis points for the data in array -> [int/double]
    ISMI_bound_est = ISMI bound data in array -> [double]
    gen_error = the generilization estimate data in array -> [double]
    """
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel("n")
    ax.set_ylabel("Error")
    marker_list = ['x', 'o', 's']
    colors = ["#f16806", "green", "#2961a6"]
    for y in range(len(y_list)):
        ax.plot(x_list, y_list[y][1], marker = marker_list[y], color = colors[y], label = y_list[y][0])
        
    ax.spines["top"].set_color('#c1c1c1')
    ax.spines["bottom"].set_color('#c1c1c1')
    ax.spines["left"].set_color('#c1c1c1')
    ax.spines["right"].set_color('#c1c1c1')
    ax.set_xticks(x_list)
    ax.set_xticklabels(x_list)
    plt.tick_params(left = False)
    plt.tick_params(bottom = False)
    ax.grid(color='#f1f1f1')
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.savefig("ExperimentResultsPlot2.png", dpi = 600)
    tikzplotlib.save(figure = fig,filepath="ExperimentResultsPlot2.tex", axis_width = "6", axis_height = "4")
    #tikz_save("ExperimentResultsPlot2.tikz")
    plt.close()
    

def write_to_csv(x_list, ISMI_bound_est, gen_error):
    workbook = xlsxwriter.Workbook('experimentResults.xlsx', {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet()
    data = [x_list, ISMI_bound_est, gen_error]
    worksheet.write(0, 0, 'Number of samples')
    worksheet.write(0, 1, 'ISMI')
    worksheet.write(0, 2, 'Gen Error')

    for column in range(3):
        for row in range(1,len(x_list)+1):
            worksheet.write(row, column, data[column][row-1])
    workbook.close()


def experiment():
    """
    Function that executes all functions needed to perform the neccessary
    calculations.  
    """
    
    # experiment parameters
    n_start = 2 
    n_stop = 36
    n_step = 2
    
    N = 50000 
    number_of_test_samples = 5000 
    mu_1 = [-1,-1]
    mu_neg_1 = [1,1]
    sigma = [[2, 0], [0, 2]]
    
    x_list = []
    ISMI_bound_est = []
    gen_error = []

    for n in range(n_start, n_stop+n_step, n_step):
        W_coefficents = []
        dataSet = []
        x_list.append(n)
        est_gen_error_list = []
        
        for _ in range(N):
            Z = create_dataset(mu_1, mu_neg_1, sigma, n)
            W_model = train_W(Z)
            
            random_index = randint(0, 2*n-1)
            dataSet.append(Z[random_index])
                
            W_coefficents.append(W_model.coef_[0,:])
            Z_test = create_dataset(mu_1, mu_neg_1, sigma, number_of_test_samples)
            est_gen_error_list.append(estimate_gen_error(W_model, Z_test, Z))
        
        ISMI_bound_est.append(sqrt(BI_KSG_estimator.est_BI_KSG(dataSet, W_coefficents) /2))
        gen_error.append(mean(est_gen_error_list))
        print("n =", n)
        print("gen error =", mean(est_gen_error_list))
        print("ISMI gen error =", ISMI_bound_est[-1])
        print("-------")

    # result from original authors
    ISMI_org = [0.284, 0.237, 0.201, 0.187, 0.166, 0.148, 0.142, 0.129, 0.134, 0.110, 0.110, 0.1, 0.095, 0.094, 0.090, 0.090, 0.09, 0.09]
    
    plot_results(x_list, [["ISMI", ISMI_org[0:len(ISMI_bound_est)]], ["ISMI rep" , ISMI_bound_est], ["Gen error", gen_error]])
    write_to_csv(x_list, ISMI_bound_est, gen_error)


if __name__ == "__main__":
    experiment()

