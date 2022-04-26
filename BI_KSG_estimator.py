"""
This program estimates the mutual information with the BI_KSG method from
https://arxiv.org/pdf/1604.03006.pdf. 

The code for the summation term in the equation
has been taken from https://github.com/wgao9/knnie.

Created by Kevin Pettersson and Reza Qorbani

15/3-22
"""


from numpy import concatenate
from scipy.spatial import cKDTree
from math import log, pi, pow, inf
from scipy.special import digamma, gamma



def est_BI_KSG(x, y, k = 5):
    """
    A function for estimating the bias-improved KSG (BI_KSG) estimator,
    according to https://arxiv.org/pdf/1604.03006.pdf. The equation is
    number 4 in the paper. The estimator estimates I{X;Y} from empirical 
    samples. Where X and Y are IID variables.

    Input:
    x = samples in 2D array -> [[...],[...]]
    y = samples in 2D array -> [[...],[...]]
    k = k nereast neighbour variable -> int

    Output:
    est_MI_BI_KSG = the estimated mutual information for the input data
    calculated with the BI_KSG method -> double
    """
    
    if(len(x) != len(y)):
        raise ValueError("X and Y must be same length!")

    N = len(x)         # number of samples
    dim_x = len(x[0])  # dimension of x
    dim_y = len(y[0])  # dimension of y
    
    data = concatenate((x, y), axis=1)

    # find nearest neighbours to all points
    # cKDTRee look up for nearest neighbours of any point
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    tree_xy = cKDTree(data)

    # find the distances 
    knn_dis = []
    for point in data:
        distances, indices = tree_xy.query(point, k+1, p=inf)
        knn_dis.append(distances[k])
    
    c_d_x =  volume_d_dimensional_unit_ball(dim_x)
    c_d_y = volume_d_dimensional_unit_ball(dim_y)
    c_d_x_y = volume_d_dimensional_unit_ball(dim_x+dim_y) # c_(d_x+d_y)
    
    # all the terms that are not within the summation are in this variable 
    non_summation_terms = digamma(k) + log(N) + log((c_d_x*c_d_y)/(c_d_x_y))/(N) # divison with N to normalize

    summation_terms = 0
    # summation part eq
    for i in range(N):
        summation_terms += log(len(tree_x.query_ball_point(x[i], knn_dis[i], p=inf)))/N # x summation term
        summation_terms += log(len(tree_y.query_ball_point(y[i], knn_dis[i], p=inf)))/N # y dummation term
        
    est_MI_BI_KSG =  non_summation_terms - summation_terms
    
    return est_MI_BI_KSG


def volume_d_dimensional_unit_ball(dimension):
    """
    Function for calculating the volume of a d-dimensional 
    unit l2 ball according to https://arxiv.org/pdf/1604.03006.pdf.

    Input:
    dimension = the diemension of the ball -> int

    Output:
    volume = if the d-dimensional unit l2 ball -> double
    """

    return (pow(pi, dimension/2)/(gamma(dimension/2+1)))