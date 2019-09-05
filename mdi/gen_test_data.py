import numpy as np

if(__name__ == "__main__"):
    dim = 30
    total_len = 10000
    normal_data = np.random.normal(0,1,total_len*dim).reshape(total_len, dim)

    anomaly_num = 1
    anomaly_ratio = 0.1
    gen_sum = int(anomaly_num/2/anomaly_ratio)
    anomaly_starts_ends = np.unique(np.random.choice(total_len,2*gen_sum, replace=False)).reshape(gen_sum, 2)
    anomaly_starts_ends = anomaly_starts_ends[np.unique(np.random.choice(gen_sum, anomaly_num, replace=False)),:]

    for i in range(anomaly_num):
        mu_i = 1.5+np.random.random()
        sigma_i = np.abs(1+np.random.normal(0,2,1))
        len_i = anomaly_starts_ends[i][1]-anomaly_starts_ends[i][0]
        normal_data[anomaly_starts_ends[i][0]:anomaly_starts_ends[i][1],:] = np.random.normal(mu_i, sigma_i, len_i*dim).reshape(len_i, dim)

    np.save("test_data.npy", normal_data)
    np.save("anomaly_intervals.npy",anomaly_starts_ends)
    print("anomaly:", anomaly_starts_ends)
