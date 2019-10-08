import os, sys
import numpy as np
import maxdiv_util
from numpy.linalg import slogdet, inv, solve
from scipy.linalg import solve_triangular, cholesky, cho_factor, cho_solve
from scipy.stats import multivariate_normal
import argparse


#
# Maximally divergent regions using Kernel Density Estimation
#
def maxdiv_parzen(K, intervals, mode='I_OMEGA', alpha=1.0, **kwargs):
    """ Scores given intervals by using Kernel Density Estimation.

    `K` is a symmetric kernel matrix whose components are K(|i - j|) for a given kernel K.

    `intervals` has to be an iterable of `(a, b, score)` tuples, which define an
    interval `[a,b)` which is suspected to be an anomaly.

    Returns: a list of `(a, b, score)` tuples. `a` and `b` are the same as in the given
             `intervals` iterable, but the scores will indicate whether a given interval
             is an anomaly or not.
    """

    # compute integral sums for each column within the kernel matrix
    K_integral = np.cumsum(K if not np.ma.isMaskedArray(K) else K.filled(0), axis=0)
    # the sum of all kernel values for each column
    # is now given in the last row
    sums_all = K_integral[-1, :]
    # n is the number of data points considered
    n = K_integral.shape[0]
    if np.ma.isMaskedArray(K):
        i = 0
        while (K.mask[i, :].all()):
            i += 1
        mask = K.mask[i, :]
        numValidSamples = n - mask.sum()
    else:
        numValidSamples = n

    # list of results
    scores = []

    # small constant to avoid problems with log(0)
    eps = 1e-7

    # indicators for points inside and outside of the anomalous region
    extreme = np.zeros(n, dtype=bool)
    non_extreme = np.ones(n, dtype=bool)
    # loop through all intervals
    for a, b, base_score in intervals:

        score = 0.0

        extreme[:] = False
        extreme[a:b] = True
        non_extreme = np.logical_not(extreme)
        if np.ma.isMaskedArray(K):
            extreme[mask] = False
            non_extreme[mask] = False

        # number of data points in the current interval
        extreme_interval_length = b - a if not np.ma.isMaskedArray(K) else b - a - mask[a:b].sum()
        # number of data points outside of the current interval
        non_extreme_points = numValidSamples - extreme_interval_length

        # compute the KL divergence
        # the mode parameter determines which KL divergence to use
        # mode == SYM does not make much sense right now for alpha != 1.0
        if mode == "IS_I_OMEGA":
            # for comments see OMEGA_I
            # this is a very experimental mode that exploits importance sampling
            sums_extreme = K_integral[b - 1, :] - (K_integral[a - 1, :] if a > 0 else 0)
            sums_non_extreme = sums_all - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points
            weights = sums_extreme / (sums_non_extreme + eps)
            weights[extreme] = 1.0
            weights /= np.sum(weights)
            kl_integrand1 = np.sum(weights * np.log(sums_non_extreme + eps))
            kl_integrand2 = np.sum(weights * np.log(sums_extreme + eps))
            negative_kl_I_Omega = alpha * kl_integrand1 - kl_integrand2
            score += - negative_kl_I_Omega

        if mode == "OMEGA_I" or mode == "SYM":
            # sum up kernel values to get non-normalized
            # kernel density estimates at single points for p_I and p_Omega
            # we use the integral sums in K_integral
            # sums_extreme and sums_non_extreme are vectors of size n
            sums_extreme = K_integral[b - 1, non_extreme] - (K_integral[a - 1, non_extreme] if a > 0 else 0)
            sums_non_extreme = sums_all[non_extreme] - sums_extreme
            # divide by the number of data points to get the final
            # parzen scores for each data point
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points

            # version for maximizing KL(p_Omega, p_I)
            # in this case we have p_Omega
            kl_integrand1 = np.mean(np.log(sums_extreme + eps))
            kl_integrand2 = np.mean(np.log(sums_non_extreme + eps))
            negative_kl_Omega_I = alpha * kl_integrand1 - kl_integrand2
            score += - negative_kl_Omega_I

        # version for maximizing KL(p_I, p_Omega)
        if mode == "I_OMEGA" or mode == "SYM":
            # for comments see OMEGA_I
            sums_extreme = K_integral[b - 1, extreme] - (K_integral[a - 1, extreme] if a > 0 else 0)
            sums_non_extreme = sums_all[extreme] - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points
            kl_integrand1 = np.mean(np.log(sums_non_extreme + eps))
            kl_integrand2 = np.mean(np.log(sums_extreme + eps))
            negative_kl_I_Omega = alpha * kl_integrand1 - kl_integrand2
            score += - negative_kl_I_Omega

        # Cross Entropy
        if mode == "CROSSENT" or mode == "CROSSENT_TS":
            sums_extreme = K_integral[b - 1, extreme] - (K_integral[a - 1, extreme] if a > 0 else 0)
            sums_non_extreme = sums_all[extreme] - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points
            score -= np.sum(np.log(sums_non_extreme + eps)) if mode == "CROSSENT_TS" else np.mean(
                np.log(sums_non_extreme + eps))

        # Jensen-Shannon Divergence
        if mode == 'JSD':
            jsd = 0.0

            # Compute p_I and p_Omega for extremal points
            sums_extreme = K_integral[b - 1, extreme] - (K_integral[a - 1, extreme] if a > 0 else 0)
            sums_non_extreme = sums_all[extreme] - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points
            # Compute (p_I + p_Omega)/2 for extremal points
            sums_combined = (sums_extreme + sums_non_extreme) / 2
            # Compute sum over extremal region
            jsd += np.mean(np.log2(sums_extreme + eps) - np.log2(sums_combined + eps))

            # Compute p_I and p_Omega for non-extremal points
            sums_extreme = K_integral[b - 1, non_extreme] - (K_integral[a - 1, non_extreme] if a > 0 else 0)
            sums_non_extreme = sums_all[non_extreme] - sums_extreme
            sums_extreme /= extreme_interval_length
            sums_non_extreme /= non_extreme_points
            # Compute (p_I + p_Omega)/2 for non-extremal points
            sums_combined = (sums_extreme + sums_non_extreme) / 2
            # Compute sum over non-extremal region
            jsd += np.mean(np.log2(sums_non_extreme + eps) - np.log2(sums_combined + eps))

            score += jsd / 2.0

        # store the score
        scores.append((a, b, score))

    return scores


#
# Maximally divergent regions using a Gaussian assumption
#
def maxdiv_gaussian_globalcov(X, intervals, mode="I_OMEGA", gaussian_mode="GLOBAL_COV", **kwargs):
    dimension, n = X.shape
    numValidSamples = n if not np.ma.isMaskedArray(X) else X[0, :].count()

    X_integral = np.cumsum(X if not np.ma.isMaskedArray(X) else X.filled(0), axis=1)
    sums_all = X_integral[:, -1]
    if (gaussian_mode == "GLOBAL_COV") and (dimension > 1):
        cov = np.ma.cov(X).filled(0)
        cov_chol = cho_factor(cov)
        logdet = slogdet(cov)[1]

    scores = []

    eps = 1e-7

    for a, b, base_score in intervals:

        extreme_interval_length = b - a if not np.ma.isMaskedArray(X) else X[0, a:b].count()
        non_extreme_points = numValidSamples - extreme_interval_length
        sums_extreme = X_integral[:, b - 1] - (X_integral[:, a - 1] if a > 0 else 0)
        sums_non_extreme = sums_all - sums_extreme
        sums_extreme /= extreme_interval_length
        sums_non_extreme /= non_extreme_points

        diff = sums_extreme - sums_non_extreme

        if (gaussian_mode == "GLOBAL_COV") and (dimension > 1):
            score = diff.T.dot(cho_solve(cov_chol, diff))
            if (mode == "CROSSENT") or (mode == "CROSSENT_TS"):
                score += slogdet
        else:
            score = np.sum(diff * diff)
        if (mode == "CROSSENT") or (mode == "CROSSENT_TS"):
            score += dimension * (1 + np.log(2 * np.pi))
        scores.append((a, b, score))

    return scores


### using log-likelihood only  score = |in|*S^{-1}.*S_in + |out|*S^{-1}.*S_out
def maxdiv_ll(X, intervals):
    n, dimension = X.shape

    X_integral = np.cumsum(X, axis=0)
    X_integral_all = X_integral[-1, :]
    mean_all = X_integral_all / n

    # X_outer = np.zeros((n, dimension, dimension))
    # for i in range(n):
    #     X_outer[i, :, :] = np.outer(X[i, :], X[i, :])
    # X_outer_integral = np.cumsum(X_outer.reshape(n, -1), axis=0).reshape(n, dimension, dimension)

    #np.save("../../data/wadi/outer_interval.npy", X_outer_integral)
    X_outer_integral = np.load('../../data/wadi/outer_interval.npy')
    X_outer_integral_all = X_outer_integral[-1, :, :]

    cov = X_outer_integral_all / n - np.outer(mean_all, mean_all)
    cov_chol = cho_factor(cov)
    cov_inv = np.linalg.inv(cov)

    scores = []

    for a, b, score in intervals:
        in_len = b - a
        out_len = n - in_len
        X_integral_in = X_integral[b - 1, :] if a == 0 else X_integral[b - 1, :] - X_integral[a - 1, :]
        X_outer_integral_in = X_outer_integral[b - 1, :, :] if a == 0 else \
            X_outer_integral[b - 1, :, :] - X_outer_integral[a - 1, :, :]
        X_integral_out = X_integral_all - X_integral_in
        X_outer_integral_out = X_outer_integral_all - X_outer_integral_in

        # params:

        mean_in = X_integral_in / in_len
        mean_out = X_integral_out / out_len

        w11 = 0.7
        w12 = 1 - w11
        w21 = 0.4
        w22 = 1 - w21

        w1 = 1
        w2 = 1

        mean_1 = w11 * mean_in + w12 * mean_out
        mean_2 = w21 * mean_in + w22 * mean_out

        val1 = X_outer_integral_in / in_len + np.outer(mean_1, mean_1)*(w11*w11-2*w11)\
               +np.outer(mean_1, mean_2)*(w11*w12-w12)+np.outer(mean_2,mean_1)*(w11*w12-w12)\
                +np.outer(mean_2, mean_2)*w12*w12
        val2 = X_outer_integral_out / out_len  + np.outer(mean_2, mean_2)*(w22*w22-2*w22)\
               +np.outer(mean_2, mean_1)*(w22*w21-w21)+np.outer(mean_1,mean_2)*(w22*w21-w21)\
                +np.outer(mean_1, mean_1)*w21*w21

        val = val1*w1+val2*w2
        score = np.sum(cov_inv*val)

        scores.append([a, b, score])

    return scores


#
# Maximally divergent regions using a Gaussian assumption
#
def maxdiv_gaussian(X, intervals, mode="I_OMEGA", gaussian_mode="COV", **kwargs):
    if (gaussian_mode in ("COV_TS", "TS")):
        gaussian_mode = "COV"
        mode = "TS"
    if (gaussian_mode != "COV"):
        return maxdiv_gaussian_globalcov(X, intervals, mode, gaussian_mode)

    dimension, n = X.shape
    numValidSamples = n if not np.ma.isMaskedArray(X) else X[0, :].count()
    X_integral = np.cumsum(X if not np.ma.isMaskedArray(X) else X.filled(0), axis=1)
    sums_all = X_integral[:, -1]
    scores = []

    outer_X = np.apply_along_axis(lambda x: np.ravel(np.outer(x, x)), 0, X)
    if (np.ma.isMaskedArray(X)):
        outer_X[:, X.mask[0, :]] = 0
    outer_X_integral = np.cumsum(outer_X, axis=1)
    outer_sums_all = outer_X_integral[:, -1]

    if (mode == "TS"):
        ts_mean = X.shape[0] + (X.shape[0] * (X.shape[0] + 1)) / 2
        ts_sd = np.sqrt(2 * ts_mean)

    eps = 1e-7
    for a, b, base_score in intervals:
        score = 0.0
        extreme_interval_length = b - a if not np.ma.isMaskedArray(X) else X[0, a:b].count()
        non_extreme_points = numValidSamples - extreme_interval_length

        sums_extreme = X_integral[:, b - 1] - (X_integral[:, a - 1] if a > 0 else 0)
        sums_non_extreme = sums_all - sums_extreme
        sums_extreme /= extreme_interval_length
        sums_non_extreme /= non_extreme_points

        outer_sums_extreme = outer_X_integral[:, b - 1] - (outer_X_integral[:, a - 1] if a > 0 else 0)
        outer_sums_non_extreme = outer_sums_all - outer_sums_extreme
        outer_sums_extreme /= extreme_interval_length
        outer_sums_non_extreme /= non_extreme_points

        cov_extreme = np.reshape(outer_sums_extreme, [dimension, dimension]) - \
                      np.outer(sums_extreme, sums_extreme) + eps * np.eye(dimension)
        cov_non_extreme = np.reshape(outer_sums_non_extreme, [dimension, dimension]) - \
                          np.outer(sums_non_extreme, sums_non_extreme) + eps * np.eye(dimension)

        if (mode != "JSD"):
            logdet_extreme = slogdet(cov_extreme)[1]
            logdet_non_extreme = slogdet(cov_non_extreme)[1]
            diff = sums_extreme - sums_non_extreme

        if (mode == "OMEGA_I" or mode == "SYM"):
            inv_cov_extreme = inv(cov_extreme)
            kl_Omega_I = np.dot(diff, np.dot(inv_cov_extreme, diff.T))
            kl_Omega_I += np.trace(np.dot(inv_cov_extreme, cov_non_extreme))
            kl_Omega_I += logdet_extreme - logdet_non_extreme - dimension
            score += kl_Omega_I
        if (mode in ("I_OMEGA", "SYM", "TS")):
            inv_cov_non_extreme = inv(cov_non_extreme)
            kl_I_Omega = np.dot(diff, np.dot(inv_cov_non_extreme, diff.T))
            kl_I_Omega += np.trace(np.dot(inv_cov_non_extreme, cov_extreme))
            kl_I_Omega += logdet_non_extreme - logdet_extreme - dimension

            if (mode == "TS"):
                score += (extreme_interval_length * kl_I_Omega - ts_mean) / ts_sd
            else:
                score += kl_I_Omega

        if (mode in ("CROSSENT", "CROSSENT_TS")):
            inv_cov_non_extreme = inv(cov_non_extreme)

            ce_I_Omega = np.dot(diff, np.dot(inv_cov_non_extreme, diff.T))
            ce_I_Omega += np.trace(np.dot(inv_cov_non_extreme, cov_extreme))
            ce_I_Omega += logdet_non_extreme + dimension * np.log(2 * np.pi)
            if (mode == "CROSSENT_TS"):
                score += (extreme_interval_length * ce_I_Omega - ts_mean) / ts_sd
            else:
                score += ce_I_Omega

        if (mode == "JSD"):
            pdf_extreme = multivariate_normal.pdf(X.T, sums_extreme, cov_extreme)
            pdf_non_extreme = multivariate_normal.pdf(X.T, sums_non_extreme, cov_non_extreme)
            pdf_combined = (pdf_extreme + pdf_non_extreme) / 2
            if (np.ma.isMaskedArray(X)):
                pdf_extreme = np.ma.MaskedArray(pdf_extreme, X.mask[0, :])
                pdf_non_extreme = np.ma.MaskedArray(pdf_non_extreme, X.mask[0, :])
                pdf_combined = np.ma.MaskedArray(pdf_combined, X.mask[0, :])

            jsd_extreme = (np.log2(pdf_extreme[a:b] + eps) - np.log2(pdf_combined[a:b] + eps)).mean()
            jsd_non_extreme = (np.log2(np.concatenate((pdf_non_extreme[:a], pdf_non_extreme[b:])) + eps) -
                               np.log2(np.concatenate((pdf_combined[:a], pdf_combined[b:])) + eps))

            score += (jsd_extreme + jsd_non_extreme) / 2.0

        scores.append((a, b, score))

    return scores


#
# Maximally divergent regions using an Ensemble of Random Projection Historgrams
#
def maxdiv_erph(X, intervals, mode="I_OMEGA", num_hist=100, num_bins=None, discount=1, **kwargs):
    dimension, n = X.shape
    numValidSamples = n if not np.ma.isMaskedArray(X) else X[0, :].count()
    scores = []

    if (num_bins is not None) and (num_bins < 1):
        num_bins = None

    if (discount < 1e-7):
        discount = 1e-7

    # Generate random projections
    proj_dims = int(round(np.sqrt(dimension)))
    dim_range = np.arange(dimension)
    proj = np.zeros((num_hist, dimension))
    for i in range(num_hist):
        np.random.shuffle(dim_range)
        proj[i, dim_range[:proj_dims]] = np.random.randn(proj_dims)

    # Project data and initialize histograms
    Xp = proj.dot(X) if not np.ma.isMaskedArray(X) else np.ma.dot(proj, x)
    hist = [Histogram1D]


class Histogram1D(object):
    def __init__(self, X, num_bins=None, store_data=True):
        object.__init__(self)
        self.vmin, self.vmax = X.min(), X.max()

        if (num_bins is None):

            max_pml = 0.0
            argmax_pml = None
            last_ll = np.ndarray((20,))
            last_pen = np.ndarray((20,))
            for b in range(2, len(X) // 2 + 1):
                self.num_bins = b
                self.fit(X)
                pml, ll, penalty = self._penaizedML()
                last_ll[b % 20] = ll
                last_pen[b % 20] = penalty
                if (argmax_pml is None) or (pml > max_pml):
                    max_pml, argmax_pml = pml, b
                elif (b >= 22) and (b - argmax_pml > 20) \
                        and (np.mean(last_ll[1:] - last_ll[:-1]) < np.mean(last_pen[1:] - last_pen[:-1])):
                    break
            self.num_bins = argmax_pml
        else:
            self.num_bins = num_bins

        if (store_data):
            self.fix(X)
        else:
            self.counts = np.zeros(self.num_bins, dtype=int)
            self.N = 0

    def fit(self, X, ind=False):
        self.counts = np.zeros(self.num_bins, dtype=int)
        self.N = len(X) if not np.ma.isMaskedArray(X) else X.count()
        if not ind:
            X = self.indices(X)
        self.counts = np.array([(X == i).sum() for i in range(self.num_bins)])

    def pdf(self, X, ind=False):
        if not ind:
            X = self.indices(X)
        prob = self.num_bins * self.counts(X).astype(float) / self.N
        if np.ma.isMaskedArray(X):
            prob = np.ma.MaskedArray(X, X.mask)
        return prob

    def indices(self, X):
        ind = ((X - self.vmin) * self.num_bins / (self.vmax - self.vmin)).astype(int)
        ind[ind < 0] = 0
        ind[ind >= self.num_bins] = self.num_bins - 1
        return ind

    def _penalizedML(self):
        ll = np.sum(self.counts.astype(float) * np.log(self.num_bins * self.counts.astype(float) / self.N + 1e-12))
        penalization = self.num_bins - 1 + np.log(self.num_bins) ** 2.5
        return ll - penalization, ll, penalization


def denseRegionProposals(N, int_min=6, int_max=50):
    # print("int_min = ", int_min )
    # print("int_max = ", int_max)
    # print("N = ",N)
    regions = []
    int_sizes = [int_min]
    if (int_min < int_max):
        int_sizes = np.unique(np.linspace(int_min, int_max, 40).astype(int))
        print("int sizes: ", int_sizes)
    for int_size in int_sizes:
        # print('int_size:', int_size)
        stride = np.max((1, int(int_size * 0.05)))
        # print("stride:", stride)
        for start in range(0, N - int_size, stride):
            # print("start : ",start)
            regions.append((int(start), int(start + int_size), 0.0))

    return regions


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


if (__name__ == "__main__"):

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wadi',
                        choices=['wadi', 'swat'])
    parser.add_argument('--dataset_path', type=str, default='../../data')
    parser.add_argument('--dataset_filename', type=str, default='_1.npy')
    parser.add_argument('--mdi_method', type=str, default='ll')
    parser.add_argument('--use_bnaf', type=int, default=1,
                        choices=[1, 0])
    args = parser.parse_args()
    args.use_bnaf = bool(args.use_bnaf)
    # load_data
    if args.use_bnaf:
        data = np.load(os.path.join(args.dataset_path, args.dataset + "/anomaly_1.npy"))
    else:
        if args.dataset == 'wadi':
            data = np.load(os.path.join(args.dataset_path, 'wadi/anomaly' + args.dataset_filename))
        elif args.dataset == 'swat':
            data = np.load(os.path.join(args.dataset_path, 'swat/anomaly' + args.dataset_filename))
        else:
            print("args.dataset error = ", args.dataset)
            data = None
    print("data.shape", data.shape)
    data_attack = np.load('../../data/wadi/attack_level.npy')

    anomaly_intervals_gth = get_attack_interval(data_attack)

    # all I need is K and intervals
    regions_proposals = denseRegionProposals(data.shape[0],
                                             int_min=np.min(anomaly_intervals_gth[:, 1] - anomaly_intervals_gth[:, 0]), \
                                             int_max=np.max(anomaly_intervals_gth[:, 1] - anomaly_intervals_gth[:, 0]))

    print("region proposals size:", len(regions_proposals))

    methods = args.mdi_method.split('+')
    for method in methods:
        if method == "parzen":
            K = maxdiv_util.calc_gaussian_kernel(data.T, normalized=False)
            print("K.size", K.shape)
            interval_scores = maxdiv_parzen(K, regions_proposals)

        elif method == "gaussian_globalcov":
            print("gaussian_globbal_cov")
            interval_scores = maxdiv_gaussian_globalcov(data.T, regions_proposals)

        elif method == "gaussian":
            print("gaussian")
            interval_scores = maxdiv_gaussian(data.T, regions_proposals)

        elif method == "ll":
            print("ll")
            interval_scores = maxdiv_ll(data, regions_proposals)

        else:
            print("method wrong")
            interval_scores = None

        if args.use_bnaf:
            bnaf_str = "bnaf"
        else:
            bnaf_str = 'nonbnaf'
        output_file_name = "region_proposals_{}_{}_{}.npy".format(
            args.dataset, bnaf_str, method)

        np.save(output_file_name, interval_scores)
