import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_attack_interval(label):
    starts = []
    ends = []

    for i in range(len(label)-1):
        if(label[i] == 0 and label[i+1]== 1):
            starts.append(i+1)
        if(label[i]==1 and label[i+1] == 0):
            ends.append(i+1)
    int_num = np.min((len(starts), len(ends)))
    return np.concatenate((starts[:int_num], ends[:int_num])).reshape(2, int_num).transpose(1,0)
def eval_measure(test, pred, test_th = 0.02, pred_th = 0.24):
    TP, FP, TN, FN = 0,0,0,0
    for i in range(len(test)):
        if(test[i] >test_th):
            if(pred[i]>pred_th):
                TP+=1
            elif(pred[i]<=pred_th):
                FN +=1
        elif(test[i]<=test_th):
            if(pred[i]>pred_th):
                FP +=1
            elif(pred[i]<=pred_th):
                TN+=1
    if(TP+FP==0):
        print("TP+FP==0")
        return (0,0,0)
    if(TP+FN==0):
        print("TP+FN==0")
        return (0,0,0)

    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    F1 = 2*pre*rec/(pre+rec)
    #print("pre: ", pre, ";  rec: ", rec, "; F1: ", F1)
    return (pre, rec, F1)

if(__name__ == "__main__"):
    # load_data
    data_name = "../../data/wadi/anomaly_pc4.npy"
    data = np.load(data_name)
    print("data.shape", data.shape)
    data_attack = data[:, -1]
    data = data[:, :-1]

    anomaly_intervals_gth = get_attack_interval(data_attack)
    print("anomaly_intervals = ", anomaly_intervals_gth)

    anomaly_level = np.zeros(data.shape[0])
    for i in range(len(anomaly_intervals_gth)):
        for j in range(anomaly_intervals_gth[i][0], anomaly_intervals_gth[i][1]):
            anomaly_level[j] = 1
    region_proposals = np.load("region_proposals_gaussian.npy")
    region_proposals_max = np.argmax(region_proposals[:,-1])
    print("max score interval:", region_proposals[region_proposals_max])


    region_proposals_df = pd.DataFrame({"start":region_proposals[:,0],\
                                        "end": region_proposals[:,1],\
                                        "score": region_proposals[:,2]})
    print(region_proposals_df.head())
    region_proposals_df = region_proposals_df.sort_values("score", ascending=False)
    print(region_proposals_df.head())
    anomaly_scores = np.zeros(len(region_proposals))
    anomaly_scores_pred = np.zeros(len(region_proposals))

    anomaly_time_scores = np.zeros(data.shape[0])
    anomaly_time_weights = np.zeros(data.shape[0])+0.1

    for i in range(len(region_proposals)):
        start, end, score = int(region_proposals[i][0]), int(region_proposals[i][1]), region_proposals[i][2]

        #print("region_proposals[i]: ", region_proposals[i])
        anomaly_scores[i] = np.mean(anomaly_level[int(start):int(end)])
        anomaly_scores_pred[i] = score

        for j in range(start, end):
            anomaly_time_scores[j] += score
            anomaly_time_weights[j] += 1

    anomaly_time_scores_aver = anomaly_time_scores/anomaly_time_weights
    print("np.corrcoeff time score:", np.corrcoef(anomaly_time_scores_aver, anomaly_level))

    for pred_th in np.linspace(np.quantile(anomaly_time_scores_aver, 0.7), np.quantile(anomaly_time_scores_aver,0.92),100):
        res = eval_measure(anomaly_level, anomaly_time_scores_aver, test_th=0.5, pred_th=pred_th)
        print("for pred_th = ", pred_th, "res = ", res)

    plt.figure(figsize=(20, 10))

    range2 = np.arange(0, data.shape[0])
    plt.plot(range2, anomaly_time_scores_aver[range2], color = "b")
    for i in range(len(anomaly_intervals_gth)):
        plt.axvspan(anomaly_intervals_gth[i][0], anomaly_intervals_gth[i][1], alpha= 0.3, color ="red")
    plt.show()
    #for i in range(data.shape[1]):
    #    plt.subplot(normal_pc.shape[1], 2, 2 * i + 1)
    #    plt.plot(range2, pred_y_data[range2, :][:, i], color="g")
    #    for j in range(len(start)):
    #        if (start[j] in range2 or end[j] in range2):
    #            plt.axvspan(start[j], end[j], alpha=0.3, color='red')
        #plt.title("pc" + str(i))
#
        #plt.subplot(normal_pc.shape[1], 2, 2 * i + 2)
        #plt.plot(range2, np.abs(anomaly_pc[range2, :][:, i] - pred_y_data[range2, :][:, i]))
        #for j in range(len(start)):
      #      if (start[j] in range2 or end[j] in range2):
      #         plt.axvspan(start[j], end[j], alpha=0.3, color='red')
       # plt.title("pc" + str(i))



