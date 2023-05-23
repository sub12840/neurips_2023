import ot
import numpy as np
from sklearn.metrics import mean_squared_error
from ..models.normal_fitters import MVGaussianMLE

def get_fair_unfair_tup(unfair, barycent):
    dist_mat = ot.dist(unfair, barycent)
    dist_mat /= dist_mat.max()

    a, b = ot.unif(unfair.shape[0]), ot.unif(unfair.shape[0])

    map_transport = ot.lp.emd(a,b,dist_mat).argmax(axis=1)

    fair = barycent[list(map_transport), :]
    return fair, unfair

def get_all_wasserstein(predictions_, sensitive_, targets_):
    preds = predictions_

    s_1_preds = preds[sensitive_, :]
    s_2_preds = preds[~sensitive_, :]
    all_preds_ord = np.concatenate([s_1_preds, s_2_preds])

    y_test_s1 = targets_[sensitive_, :]
    y_test_s2 = targets_[~sensitive_, :]
    all_targets_ord = np.concatenate([y_test_s1, y_test_s2])

    dist_mat = ot.dist(s_1_preds, s_2_preds)
    dist_mat /= dist_mat.max()

    a, b = ot.unif(s_1_preds.shape[0]), ot.unif(s_2_preds.shape[0])

    dist_unfair = ot.sliced_wasserstein_distance(s_1_preds, s_2_preds, a, b, n_projections=2500)

    all_barys = []
    # Need to downsample as matrix gets too large
    for sample_ in range(13):

        downsamples = 2000
        small_s1 = s_1_preds[np.random.choice(range(s_1_preds.shape[0]), downsamples), :]
        small_s2 = s_2_preds[np.random.choice(range(s_2_preds.shape[0]), downsamples), :]

        k = small_s1.shape[0]  
        X_init = np.random.normal(0., 1., small_s1.shape)
        b2 = np.ones((k,)) / k 
        a, b = ot.unif(small_s1.shape[0]), ot.unif(small_s2.shape[0])
        a, b = ot.unif(small_s1.shape[0]), ot.unif(small_s2.shape[0])

        bary_obs_1 = ot.lp.free_support_barycenter([small_s1, small_s2],
                                                    [a,b],
                                                    X_init,
                                                    b2)
        
        all_barys.append(bary_obs_1)

    bary_obs_samples=np.vstack(all_barys)

    target_1 = bary_obs_samples[np.random.choice(range(bary_obs_samples.shape[0]), s_1_preds.shape[0]), :]
    target_2 = bary_obs_samples[np.random.choice(range(bary_obs_samples.shape[0]), s_2_preds.shape[0]), :]

    fair_1, unfair_1 = get_fair_unfair_tup(s_1_preds, target_1)
    fair_2, unfair_2 = get_fair_unfair_tup(s_2_preds, target_2)
    all_fair_ord = np.concatenate([fair_1, fair_2])

    interpolated_1 = 0.5*fair_1 + 0.5*unfair_1
    interpolated_2 = 0.5*fair_2 + 0.5*unfair_2
    all_interplated_ord = np.concatenate([interpolated_1, interpolated_2])

    a, b = ot.unif(s_1_preds.shape[0]), ot.unif(s_2_preds.shape[0])

    dist_semifair = ot.sliced_wasserstein_distance(interpolated_1, interpolated_2, a, b, n_projections=2500)
    dist_fair = ot.sliced_wasserstein_distance(fair_1, fair_2, a, b, n_projections=2500)

    # Get budget
    marg_1, marg_2 = preds.mean(axis=0)
    multiplication_factor = marg_2/marg_1

    total_budget_fair = (np.concatenate([fair_1, fair_2]).mean(axis=0) * np.array([multiplication_factor, 1])).sum()
    total_budget_unfair = (preds.mean(axis=0) * np.array([multiplication_factor, 1])).sum()
    total_budget_interpo = (np.concatenate([interpolated_1, interpolated_2]).mean(axis=0) * np.array([multiplication_factor, 1])).sum()

    # MSE
    mse_1_unfair = mean_squared_error(all_preds_ord[:, 0], all_targets_ord[:, 0])
    mse_2_unfair = mean_squared_error(all_preds_ord[:, 1], all_targets_ord[:, 1])

    mse_1_fair = mean_squared_error(all_fair_ord[:,0], all_targets_ord[:, 0])
    mse_2_fair = mean_squared_error(all_fair_ord[:,1], all_targets_ord[:, 1])

    mse_1_semifair = mean_squared_error(all_interplated_ord[:,0], all_targets_ord[:, 0])
    mse_2_semifair = mean_squared_error(all_interplated_ord[:,1], all_targets_ord[:, 1])

    result_dict = {
        'dist_unfair': dist_unfair, 
        'dist_semifair': dist_semifair, 
        'dist_fair': dist_fair, 

        'budget_unfair': total_budget_unfair, 
        'budget_semifair': total_budget_interpo, 
        'budget_fair': total_budget_fair,

        'mse_1_unfair': mse_1_unfair,
        'mse_2_unfair': mse_2_unfair,
        'mse_1_semifair': mse_1_semifair,
        'mse_2_semifair': mse_2_semifair,
        'mse_1_fair': mse_1_fair,
        'mse_2_fair': mse_2_fair,

    }
    return result_dict



def get_all_wasserstein_param(predictions_, sensitive_, targets_):
    preds = predictions_

    s_1_preds = preds[sensitive_, :]
    s_2_preds = preds[~sensitive_, :]
    all_preds_ord = np.concatenate([s_1_preds, s_2_preds])

    y_test_s1 = targets_[sensitive_, :]
    y_test_s2 = targets_[~sensitive_, :]
    all_targets_ord = np.concatenate([y_test_s1, y_test_s2])

    dist_mat = ot.dist(s_1_preds, s_2_preds)
    dist_mat /= dist_mat.max()

    all_barys = []
    for sample_ in range(13):

        downsamples = 2000
        small_s1 = s_1_preds[np.random.choice(range(s_1_preds.shape[0]), downsamples), :]
        small_s2 = s_2_preds[np.random.choice(range(s_2_preds.shape[0]), downsamples), :]

        k = small_s1.shape[0]  
        X_init = np.random.normal(0., 1., small_s1.shape)
        b2 = np.ones((k,)) / k 
        a, b = ot.unif(small_s1.shape[0]), ot.unif(small_s2.shape[0])

        bary_obs_1 = ot.lp.free_support_barycenter([small_s1, small_s2],
                                                    [a,b],
                                                    X_init,
                                                    b2)
        
        all_barys.append(bary_obs_1)

    bary_obs_samples=np.vstack(all_barys)

    mvnormfitter = MVGaussianMLE()
    mvnormfitter.fit(bary_obs_samples)

    samples_1_mvnorm = mvnormfitter.sample(s_1_preds.shape[0])
    samples_2_mvnorm = mvnormfitter.sample(s_2_preds.shape[0])

    eps=0.5
    new_vals_1 = eps*s_1_preds + (1-eps)*samples_1_mvnorm
    new_vals_2 = eps*s_2_preds + (1-eps)*samples_2_mvnorm
    all_param_semifair = np.concatenate([new_vals_1, new_vals_2])
    
    all_param_fair = np.concatenate([samples_1_mvnorm,
                                     samples_2_mvnorm])


    dist_param_semifair = ot.sliced_wasserstein_distance(new_vals_1, new_vals_2)
    dist_param_fair = ot.sliced_wasserstein_distance(samples_1_mvnorm, samples_2_mvnorm)


    result_dict = {
        'dist_param_semifair': dist_param_semifair, 
        'dist_param_fair': dist_param_fair,
        'bary_chek': bary_obs_samples,
        'samp_1': samples_1_mvnorm, 
        'samp_2': samples_2_mvnorm,
        'estim': mvnormfitter

    }
    return result_dict
