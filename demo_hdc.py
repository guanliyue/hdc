import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import sys
from sqlalchemy.sql.expression import except_

sys.path.append("..")


# from cluster_utils import cluster_index

def mmeans(X, k, maxIt):
    global final_cu_cen
    global original_dataset
    global iterations
    numPoints, numDim = X.shape
    dataSet = np.zeros((numPoints, numDim + 2))
    dataSet[:, :-2] = X

    #     aa = np.random.uniform(1,k+1,size = (3))

    # for the example in Fig. 3
    iniCentroids = readDataSet("./ini.csv", False)

    dataSet[:, -1] = iniCentroids[:, -1]
    ## end centroid from file(not random) ##

    #     dataSet[-1,-2] = -100
    #     dataSet[-2,-2] = -100
    #     dataSet[-3,-2] = -100
    #     dataSet[-4,-2] = -100
    cu_cen = getCentroids(dataSet, k)
    #     print(cu_cen)

    #     cu_cen[1,0] = cu_cen[1,0] -3
    #     cu_cen[1,1] = cu_cen[1,1] -1.5
    #     cu_cen = dataSet[np.random.randint(numPoints, size = k), :]
    #     cu_cen[:, -1] = range(1, k + 1)

    # plot ##
    colorPool = ['red', '#680184', 'pink', 'yellow', 'green']
    subPlotNum = 0
    subPlotNum = subPlotNum + 1
    plt.subplot(6, 4, subPlotNum)
    ## num of line represents num of clustering
    plt.scatter(dataSet[dataSet[:, -1] == 1, 0], dataSet[dataSet[:, -1] == 1, 1], c=colorPool[0])
    plt.scatter(dataSet[dataSet[:, -1] == 2, 0], dataSet[dataSet[:, -1] == 2, 1], c=colorPool[1])
    #     plt.scatter(dataSet[dataSet[:, -1] == 3, 0], dataSet[dataSet[:, -1] == 3, 1], c=colorPool[2])
    # # plt.scatter(dataSet[dataSet[:, 2] == 4, 0], dataSet[dataSet[:, 2] == 4, 1], c=colorPool[3])
    # # plt.scatter(dataSet[dataSet[:, 2] == 5, 0], dataSet[dataSet[:, 2] == 5, 1], c=colorPool[4])
    plt.scatter(cu_cen[0, 0], cu_cen[0, 1], c=colorPool[0], marker='*', s=100)
    plt.scatter(cu_cen[1, 0], cu_cen[1, 1], c=colorPool[1], marker='*', s=100)
    #     plt.scatter(cu_cen[2, 0], cu_cen[2, 1], c=colorPool[2], marker='*', s=100)
    # plt.scatter(cu_cen[3, 0], cu_cen[3, 1], c=colorPool[3], marker='*', s=100)
    # plt.scatter(cu_cen[4, 0], cu_cen[4, 1], c=colorPool[4], marker='*', s=100)
    ## end plot ##

    iterations = 0
    oldMatrix = None
    # print ('iteration: ',end='')
    write_csv(dataSet, './' + 'iter_' + str(0) + '.csv')
    while not shouldStop(oldMatrix, cu_cen, iterations, maxIt):
        # print(iterations,end=',')
        # # print 'dataSet: \n', dataSet
        # print 'cu_cen: \n', cu_cen

        oldMatrix = np.copy(cu_cen)
        iterations += 1

        dataSet = updateLabels(dataSet, original_dataset, oldMatrix)
        write_csv(dataSet, './' + 'iter_' + str(iterations) + '.csv')
        #         print(dataSet[16, -1])
        #         dataSet[17,-1] = 2
        #         dataSet[16,-1] = 2
        #         dataSet[16,-2] = 0
        #         dataSet[0,-1] =
        #         dataSet[0,-2] = 0
        dataSet = dataSet[dataSet[:, -1] != -1, :]
        original_dataset = original_dataset[original_dataset[:, -1] != -1, :]
        cu_cen = getCentroids(dataSet, k)
        #         print(iterations, cu_cen)
        final_cu_cen = copy.deepcopy(cu_cen)

        ## plot ##
        #         plt.scatter(dataSet[dataSet[:, -2] == -100, 0], dataSet[dataSet[:, -2] == -100, 1], c='black', marker='x', s=100)
        subPlotNum = subPlotNum + 1
        plt.subplot(6, 4, subPlotNum)
        #         plt.scatter(dataSet[:, 0], dataSet[:, 1], c=dataSet[:, 2])
        #         plt.scatter(cu_cen[:, 0], cu_cen[:, 1], marker='*', s=100)
        plt.scatter(dataSet[dataSet[:, -1] == 1, 0], dataSet[dataSet[:, -1] == 1, 1], c=colorPool[0])
        plt.scatter(dataSet[dataSet[:, -1] == 2, 0], dataSet[dataSet[:, -1] == 2, 1], c=colorPool[1])
        plt.scatter(dataSet[dataSet[:, -2] == -100, 0], dataSet[dataSet[:, -2] == -100, 1], c='black', marker='x',
                    s=100)
        #         plt.scatter(dataSet[dataSet[:, -1] == 3, 0], dataSet[dataSet[:, -1] == 3, 1], c=colorPool[2])
        # plt.scatter(dataSet[dataSet[:, 2] == 4, 0], dataSet[dataSet[:, 2] == 4, 1], c=colorPool[3])
        # # plt.scatter(dataSet[dataSet[:, 2] == 5, 0], dataSet[dataSet[:, 2] == 5, 1], c=colorPool[4])
        plt.scatter(cu_cen[0, 0], cu_cen[0, 1], c=colorPool[0], marker='*', s=100)
        plt.scatter(cu_cen[1, 0], cu_cen[1, 1], c=colorPool[1], marker='*', s=100)
    #         plt.scatter(cu_cen[2, 0], cu_cen[2, 1], c=colorPool[2], marker='*', s=100)
    # plt.scatter(cu_cen[3, 0], cu_cen[3, 1], c=colorPool[3], marker='*', s=100)
    # plt.scatter(cu_cen[4, 0], cu_cen[4, 1], c=colorPool[4], marker='*', s=100)
    ## end plot ##

    # np.savetxt("DS"+str(iterations), dataSet, delimiter=',')
    return dataSet


def shouldStop(oldMatrix, cu_cen, iterations, maxIt):
    if iterations > maxIt:
        return True
    return np.array_equal(oldMatrix, cu_cen)


def updateLabels(dataSet, original_dataset, oldMatrix):
    global remain_original_points
    numPoint, numDim = dataSet.shape

    dataSet_cop_up = copy.deepcopy(dataSet)
    for i in range(0, numPoint):
        gotten_label, gotten_label_mem = getLabelFromMem(dataSet, dataSet[i, :], oldMatrix)
        dataSet_cop_up[i, -1] = gotten_label
        dataSet_cop_up[i, -2] = gotten_label_mem
        # if gotten_label == -1:
        #     original_dataset[i, -1] = -1
        #     remain_original_points.append(original_dataset[i])
    ## noise handle
    # np.savetxt("dataSet-tt.csv", dataSet, delimiter=',')
    # exit(0)
    dataSet = copy.deepcopy(dataSet_cop_up)
    # print('update dataset:',dataSet)
    centroidSet_temp = dataSet[:, -1]
    centroidSet_temp = set(centroidSet_temp)
    centroid_mean_std = []
    for eachCentroid_temp in centroidSet_temp:
        # currentCluster_temp = dataSet[dataSet[:, -1] == eachCentroid_temp, :-2]
        currentCluster_temp = dataSet[dataSet[:, -1] == eachCentroid_temp, -2]
        currentCluster_temp_mean = np.mean(currentCluster_temp, axis=0)
        # currentCluster_temp_std = np.std(currentCluster_temp)
        # currentCluster_temp_theta = currentCluster_temp_mean - 2.3 * currentCluster_temp_std
        # currentCluster_temp_theta2 = currentCluster_temp_mean + 2.3 * currentCluster_temp_std
        # centroid_mean_std.append([eachCentroid_temp, currentCluster_temp_mean, currentCluster_temp_std,currentCluster_temp_theta,currentCluster_temp_theta2])
        centroid_mean_std.append([eachCentroid_temp, currentCluster_temp_mean])
        centroid_dic = {k1: v for [k1, v] in centroid_mean_std}
    allpoint_and_centroid_dis = []
    for i in range(0, numPoint):
        # print(np.linalg.norm(dataSet[i,:-2]-centroid_dic[dataSet[i, -1]]))
        allpoint_and_centroid_dis.append(dataSet[i, -2])

    # print(allpoint_and_centroid_dis)
    # exit(0)
    allpoint_and_centroid_dis = np.array(allpoint_and_centroid_dis)
    centroid_mean_std_for_near = []
    for eachCentroid_temp in centroidSet_temp:
        currentCluster_temp = allpoint_and_centroid_dis[dataSet[:, -1] == eachCentroid_temp]
        currentCluster_temp_mean = np.mean(currentCluster_temp)
        currentCluster_temp_std = np.std(currentCluster_temp)
        # currentCluster_temp_theta = currentCluster_temp_mean - global_beta_near * currentCluster_temp_std
        currentCluster_temp_theta2 = currentCluster_temp_mean + currentCluster_temp_std
        centroid_mean_std_for_near.append(
            [eachCentroid_temp, currentCluster_temp_mean, currentCluster_temp_std, currentCluster_temp_theta2])
        # centroid_mean_std_for_near.append([eachCentroid_temp, currentCluster_temp_mean])
        centroid_dic = {k1: [k1, k2, k3, v] for [k1, k2, k3, v] in centroid_mean_std_for_near}
    #     print('start')
    for i in range(0, numPoint):
        # print('numPoint: ', i, dataSet[i,:-2], allpoint_and_centroid_dis[i] ,centroid_dic[dataSet[i, -1]][0],oldMatrix[np.int(centroid_dic[dataSet[i, -1]][0])-1],centroid_dic[dataSet[i, -1]][1],centroid_dic[dataSet[i, -1]][2],centroid_dic[dataSet[i, -1]][3])
        #             dataSet[i, -1] = -1
        #         if dataSet[i, 0] == 12.8:
        #             print('lll')
        if allpoint_and_centroid_dis[i] > centroid_dic[dataSet[i, -1]][3]:
            #             print('Outliers: ', dataSet[i,:-2], allpoint_and_centroid_dis[i] ,centroid_dic[dataSet[i, -1]][0],oldMatrix[np.int(centroid_dic[dataSet[i, -1]][0])-1],centroid_dic[dataSet[i, -1]][1],centroid_dic[dataSet[i, -1]][2],centroid_dic[dataSet[i, -1]][3])
            #             dataSet[i, -1] = -1
            dataSet[i, -2] = -100
    #             original_dataset[i, -1] = -1
    #             remain_original_points.append(original_dataset[i])
    #             remain_points.append(dataSet[i])
    #             print('unnormal',dataSet[i])

    #     print('end')

    return dataSet
    # for each_noist_cluster in


def getLabelFromMem(dataSet, dataSetRow, oldMatrix):
    #     label = cu_cen[0, -1]
    #     minMem = getMembership(dataSet, dataSetRow, currentCentroid,cu_cen[0, :])
    global remain_points
    label = -1
    minMem = 100
    r2_div_r1 = 0
    r_return_r1 = 0
    r_return_r2 = 0
    centroidSet = dataSet[:, -1]
    centroidSet = set(centroidSet)
    if -1 in centroidSet:
        centroidSet.remove(-1)
    max_mem_list = []
    for iCentroidSet in centroidSet:
        mem, is_noise, t_r2_div_r1, t_return_r1, t_return_r2 = getMembership(dataSet, dataSetRow, iCentroidSet,
                                                                             oldMatrix)
        max_mem_list.append(is_noise)
        if mem < minMem:
            minMem = mem
            label = iCentroidSet
            r2_div_r1 = t_r2_div_r1
            r_return_r1 = t_return_r1
            r_return_r2 = t_return_r2
    #     print 'minMem:', minMem

    #
    #     if sum(max_mem_list) == -1 * len(max_mem_list):
    #         label = -1
    #         print('noise point:', dataSetRow)
    #         remain_points.append(dataSetRow)
    #     print(dataSetRow, r_return_r2, r_return_r1, r2_div_r1)
    return label, minMem


def getMembership(dataSet, dataSetRow, centroid, oldMatrix):
    # ## just for a test
    # if dataSetRow[0] > 0 and dataSetRow[1] > 0:
    #     print("more")
    # if dataSetRow[0] < 0 and dataSetRow[1] > 0:
    #     print("more")
    # if dataSetRow[0] > 0 and dataSetRow[1] < 0:
    #     print("more")
    # if dataSetRow[0] < 0 and dataSetRow[1] < 0:
    #     print("more")
    #
    # ## end just for a test
    # global pointDis
    # global global_alpha
    # global global_beta
    # global iterations

    # if dataSetRow[0] == 5.5:
    #     print('00000')

    currentDataRowR1etc = []
    r1 = 0
    r2 = 0
    return_r1 = 1
    return_r2 = 0
    return_r1_tmp = []
    return_r2_tmp = []
    #     centroidSet = dataSet[:, -1]
    centroidSet = np.array([centroid])
    centroidSet = set(centroidSet)
    if centroid in centroidSet:
        #         centroidSet.remove(centroid)
        r_num = len(centroidSet)
        #         print('len(centroidSet):',len(centroidSet))
        r_sum = 0
        is_noise = 1
        for eachCentroid in centroidSet:
            #             r_num = r_num + 1
            r = 0
            maxPoint = []
            allMaxPointR1_R2_Tempr_r = []
            tmp_dataSet = dataSet[dataSet[:, -2] != -100, :]
            currentCluster = tmp_dataSet[tmp_dataSet[:, -1] == eachCentroid, :]
            #             currentCluster = dataSet[dataSet[:, -1] == eachCentroid, :]

            for i in range(0, currentCluster.shape[0]):
                maxPointR1_R2_Tempr_r = []
                #                 if (1 - getMatrix(dataSet, dataSetRow[:-1], currentCluster[i, :-1])) == 0:
                #                     print '000000'
                #                 if dataSetRow[-2] != -100:
                #                     temp_getsimi = getMatrix(tmp_dataSet, dataSetRow[:-2], currentCluster[i, :-2])
                #                 else:
                temp_getsimi = getMatrix(dataSet, dataSetRow[:-2], currentCluster[i, :-2])
                r1 = 1 - temp_getsimi[0]
                #                 r1 = temp_getsimi[0]
                #                 if r1 == 1:
                #                     print('r1=1')
                #                     temp_getsimi = getMatrix(dataSet, dataSetRow[:-2], currentCluster[i, :-2])
                r1_temp = temp_getsimi[1]
                #             print currentCluster[:,:-1]
                #                 currentClusterCentroid = np.mean(currentCluster[:,:-2], axis = 0)
                currentClusterCentroid = oldMatrix[oldMatrix[:, -1] == eachCentroid, :-1][0]
                # print(currentClusterCentroid)
                #                 if dataSetRow[-2] != -100:
                #                     temp_getsimi = getMatrix(tmp_dataSet, currentClusterCentroid, currentCluster[i, :-2])
                #                 else:
                temp_getsimi = getMatrix(dataSet, currentClusterCentroid, currentCluster[i, :-2])
                r2 = temp_getsimi[0]
                r2_temp = temp_getsimi[1]
                #             print 'r1:', r1
                #             print 'r2:', r2
                tempr = 2 * r1 * r2 / (r1 + r2)
                #                 print('r1,r2,tempr:',dataSetRow,currentCluster[i, :-2],currentClusterCentroid,r1,r2,tempr)
                # tempr = r1
                # tempr = (r1 + r2) / 2
                if tempr > r:
                    return_r1 = r1
                    return_r2 = r2
                    return_r1_tmp = r1_temp
                    return_r2_tmp = r2_temp
                    #                     print('centroid:', currentClusterCentroid)
                    #                     print('r2', r2)
                    #                     print('cur_cluster:', currentCluster[i])
                    r = tempr
                    maxPoint.append(currentCluster[i])
                    maxPointR1_R2_Tempr_r.append(r1)
                    maxPointR1_R2_Tempr_r.append(r2)
                    maxPointR1_R2_Tempr_r.append(tempr)
                    maxPointR1_R2_Tempr_r.append(r)
                    maxPointR1_R2_Tempr_r.append(currentClusterCentroid)
                    allMaxPointR1_R2_Tempr_r.append(maxPointR1_R2_Tempr_r)

            try:
                maxPoint = maxPoint[len(maxPoint) - 1]
            except:
                print('a')
            # allMaxPointR1_R2_Tempr_r = allMaxPointR1_R2_Tempr_r[len(allMaxPointR1_R2_Tempr_r) - 1]
            currentDataRowR1etc = [dataSetRow, eachCentroid, maxPoint, allMaxPointR1_R2_Tempr_r]

            ## preprocess
            # tempPointDis = [currentDataRowR1etc[0][0], currentDataRowR1etc[0][1], currentDataRowR1etc[0][2],
            #                 currentDataRowR1etc[1], currentDataRowR1etc[2][0], currentDataRowR1etc[2][1], currentDataRowR1etc[2][2],
            #                 currentDataRowR1etc[3][0], currentDataRowR1etc[3][1], currentDataRowR1etc[3][2],
            #                 currentDataRowR1etc[3][3], currentDataRowR1etc[3][4][0], currentDataRowR1etc[3][4][1]]
            #
            # pointDis.append(tempPointDis)
            #             print(r)
            har_r = r
            is_noise = is_noise * r
            # if return_r1 !=0:
            #     if return_r2 / return_r1 < global_alpha:
            #         # print('use_alpha--------------------------------------------------')
            #         r = return_r1
            #             if dataSetRow[-2] == -100:
            #                 print('unnormal,use r = ', r)
            #                 r = return_r1
            r = return_r1
            is_noise = is_noise * r
            r_sum = r_sum + r

        r_sum = r_sum / r_num

    #         curr_dataset_row = dataSetRow[:-1]
    #         if curr_dataset_row[0] == 8:
    #             print('is_noise', is_noise)
    #         if is_noise > global_beta:
    # #             print('is noise:', curr_dataset_row[0])
    #             return r_sum, -1
    #     if 1 - r_sum < 0.1:
    #     print (r_sum)
    #     print('middle: ',centroid, r_sum, dataSetRow, maxPoint, return_r2/return_r1, return_r1, return_r2, r,har_r)
    #     print('---',dataSetRow, maxPoint, centroid, return_r1_tmp, return_r2_tmp, return_r1, return_r2, r, r_sum)
    return r_sum, 1, return_r2 / return_r1, return_r1, return_r2
def getMatrix(dataSet, instanceX, instanceY):
    Matrix = dataSet[:, :-2]
    sv = np.std(Matrix, axis=0, ddof=1)
    temp = np.zeros(instanceX.shape[0])
    temp[0] = np.e ** (-((instanceX[0] - instanceY[0]) ** 2
                         / (2 * sv[0] ** 1)))
    #     max_of_col = np.max(simiMatrix,axis=0)
    #     min_of_col = np.min(simiMatrix,axis=0)
    #     temp = np.zeros(instanceX.shape[0])
    #     for k in range(instanceX.shape[0]):
    # #         print '-----', (instanceX[k]-instanceY[k]+sv[k])/std[k], '====',(instanceY[k]-instanceX[k]+sv[k])/sv[k]
    # #         temp[k] = 1 - np.abs(instanceX[k]-instanceY[k]) / np.abs(max_of_col[k] - min_of_col[k])
    # #         temp[k] = max(min(((instanceX[k]-instanceY[k]+sv[k])/sv[k]),((instanceY[k]-instanceX[k]+sv[k])/sv[k])),0)
    # #
    temp[1] = np.e ** (
        -((instanceX[1] - instanceY[1]) ** 2 / (2 *
                            sv[1] ** 2)))
    return get_t_L(temp, sv)
def get_t_L(temp, std):  # Luka Tnorm
    return_a = []
    a = temp[0]
    return_a.append(std)
    return_a.append(a)
    #     a = np.mean(temp)
    for i in range(len(temp)):
        if i + 1 != len(temp):
            return_a.append(temp[i + 1])
            a = a * temp[i + 1]
    # #
    #             a = min(a, temp[i+1])
    #             a = max(0,a + temp[i+1] - 1)
    return a, return_a




def getCentroids(dataSet, k):
    cen_dataset = copy.deepcopy(dataSet)
    cen_dataset = cen_dataset[cen_dataset[:, -2] != -100]
    result = np.zeros((k, cen_dataset.shape[1]))
    for i in range(1, k + 1):
        oneCluster = cen_dataset[cen_dataset[:, -1] == i, :-2]
        result[i - 1, :-2] = np.mean(oneCluster, axis=0)
        result[i - 1, -1] = i
    return np.hstack((result[:, 0:-2], result[:, -1].reshape((result.shape[0], 1))))


def readDataSet(position, deletelabel="False"):
    starttime = time.time()
    np.set_printoptions(suppress=True, linewidth=300)
    data = np.loadtxt(position, dtype=float, delimiter=',')
    if deletelabel == 1:
        data = np.delete(data, -1, axis=1)
    endtime = time.time()
    # print("read", endtime - starttime)
    return data


def write_csv(sql_data, file_name):
    save = pd.DataFrame(sql_data)
    #      try:
    save.to_csv(file_name, header=None, index=None)


def create_folder(cu_data_dic):
    file_path_check = './01_all_entire_result/' + cu_data_dic[0:-4]
    if not os.path.exists(file_path_check):
        os.mkdir(file_path_check)


def cal_remain_point_label(each_remain_point, each_final_centroid, p_3):
    #     print(each_remain_point[:-1])
    #     print(each_remain_point[:-1]-each_final_centroid[:-1])
    distance_of_two_points = np.linalg.norm(each_remain_point[:-2] - each_final_centroid[:-1])
    return 1 / distance_of_two_points


remain_points = []
remain_original_points = []
final_cu_cen = []
pointDis = []
original_dataset = []

ddd = readDataSet("./example_Fig2.csv", False)

ddd_label = ddd[:, -1]
ddd_label_num = len(set(ddd_label))

original_dataset = copy.deepcopy(ddd[:, :-1])
last_colm1_original_dataset = [1 for _ in range(original_dataset.shape[0])]
last_colm1_original_dataset = np.array(last_colm1_original_dataset)
last_colm1_original_dataset = last_colm1_original_dataset.reshape((last_colm1_original_dataset.shape[0], 1))
#             original_dataset = np.hstack((original_dataset, last_colm1_original_dataset))
original_dataset = np.hstack((ddd[:, :-1], ddd_label.reshape((ddd_label.shape[0], 1)), last_colm1_original_dataset))
#             principalComponents = principalComponents[:,:-1]
#             principalComponents = principalComponents.reshape((principalComponents.shape[0],1))
result = mmeans(ddd[:, :-1], 2, 10)

use_for_min_dis = result
result = np.hstack((result[:, 0:-2], result[:, -1].reshape((result.shape[0], 1))))

### assign the remain_points ##
#             print(remain_points)
remain_points_and_label = []
for each_remain_point in remain_points:
    temp_remain_points_and_label = []
    assign_flag = -1
    assign_flag_simi = 0
    for each_final_centroid in final_cu_cen:
        each_label_simi = cal_remain_point_label_new(use_for_min_dis, each_remain_point, each_final_centroid[-1])
        if each_label_simi > assign_flag_simi:
            assign_flag = each_final_centroid[-1]
            assign_flag_simi = each_label_simi
    for _ in range(each_remain_point.shape[0] - 2):
        temp_remain_points_and_label.append(each_remain_point[_])
    temp_remain_points_and_label.append(assign_flag)
    remain_points_and_label.append(temp_remain_points_and_label)
remain_points_and_label = np.array(remain_points_and_label)
#             print('remain:',remain_points_and_label)
### todo
### combining the result and remain_points
combin_result = copy.deepcopy(result)
if remain_points_and_label.shape[0] != 0:
    combin_result = np.vstack((result, remain_points_and_label))
#  hstack(result, np.array(remain_points)
remain_original_points = np.array(remain_original_points)
true_original_dataset = copy.deepcopy(original_dataset[:, :-1])
#                 print('remain_points: ', remain_original_points)
if remain_original_points.shape[0] != 0:
    true_original_dataset = np.vstack((original_dataset[:, :-1], remain_original_points[:, :-1]))
#             print(true_original_dataset)
t_nmi = metrics.normalized_mutual_info_score(combin_result[:, -1], true_original_dataset[:, -1])
print('nmi_value: ', t_nmi)
colorPool = ['#440154', '#21918C', '#FDE725', '#5EC864']
temp_y = [1 for _ in range(combin_result[combin_result[:, 1] == 1, 0].shape[0])]
temp_y = np.array(temp_y)
temp_y1 = [1 for _ in range(combin_result[combin_result[:, 1] == 2, 0].shape[0])]
temp_y1 = np.array(temp_y1)
temp_y2 = [1 for _ in range(combin_result[combin_result[:, 1] == 3, 0].shape[0])]
temp_y2 = np.array(temp_y2)

plt.show()


