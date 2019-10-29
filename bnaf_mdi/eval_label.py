import numpy as np
import matplotlib.pyplot as plt
def get_attack_interval(label):
    starts = []
    ends = []

    for i in range(len(label) - 1):
        if (label[i] == 0 and label[i + 1] == 1):
            starts.append(i + 1)
        if (label[i] == 1 and label[i + 1] == 0):
            ends.append(i + 1)
    int_num = np.min((len(starts), len(ends)))
    return np.concatenate((starts[:int_num], ends[:int_num])).reshape(2, int_num).transpose(1, 0)


def eval_measure(test, pred, test_th=0.02, pred_th=0.24):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(test)):
        if (test[i] > test_th):
            if (pred[i] > pred_th):
                TP += 1
            elif (pred[i] <= pred_th):
                FN += 1
        elif (test[i] <= test_th):
            if (pred[i] > pred_th):
                FP += 1
            elif (pred[i] <= pred_th):
                TN += 1
    if (TP + FP == 0):
        print("TP+FP==0")
        return (0, 0, 0)
    if (TP + FN == 0):
        print("TP+FN==0")
        return (0, 0, 0)

    pre = TP / (TP + FP)
    rec = TP / (TP + FN)
    F1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    # print("pre: ", pre, ";  rec: ", rec, "; F1: ", F1)
    return (pre, rec, F1)

if __name__=="__main__":
    file_name = "_px.npy"
    anomaly_pred = np.load(file_name)
    anomaly_level = np.load('../../data/wadi/attack_level.npy')
    anomaly_intervals_gth = get_attack_interval(anomaly_level)
    
    #print("np.corrcoeff time score:", np.corrcoef(anomaly_pred, anomaly_level))

    max_f1 = 0
    max_th = 0
    for pred_th in np.linspace(np.quantile(anomaly_pred, 0.80), np.quantile(anomaly_pred, 0.98),
                               100):
        res = eval_measure(anomaly_level, anomaly_pred, test_th=0.5, pred_th=pred_th)
        #print("for pred_th = ", pred_th, "res = ", res)
        if res[2]>max_f1:
            max_f1=res[2]
            max_th = pred_th

    print("max_f1:", max_f1, " th=", max_th)
    plt.figure(figsize=(20, 10))

    range2 = np.arange(0, anomaly_pred.shape[0])
    plt.plot(range2, anomaly_pred[range2], color="b")
    for i in range(len(anomaly_intervals_gth)):
        plt.axvspan(anomaly_intervals_gth[i][0], anomaly_intervals_gth[i][1], alpha=0.3, color="red")
    #plt.title('{}_{}_{}.score'.format(args.dataset, 'bnaf' if args.use_bnaf else 'nonbnaf', args.mdi_method))
    plt.title(file_name)
    #plt.show()