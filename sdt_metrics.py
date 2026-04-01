import numpy as np
import pandas as pd
from scipy.stats import norm
from metadpy.mle import metad
from sklearn.metrics import roc_auc_score

def compute_sdt(actually_correct_list, ratings_eight):
    """This method computes the Type-2 SDT (Self-Determination Theory) metrics using the data collected from model answer/delgation confidence intervals
       The eight point scale goes from 
       1 = certainly delegate
       8 = certainly answer"""
    
    correct = np.array(actually_correct_list, dtype = bool)
    ratings = np.array(ratings_eight, dtype = bool)
    
    n_ratings = 8

    nR_S1 = np.zeros(n_ratings)
    nR_S2 = np.zeros(n_ratings)

    for r in range(n_ratings):
        nR_S1[r] = np.sum((~correct) & (ratings == r + 1))
        nR_S2[r] = np.sum((correct) & (ratings == r + 1))
    
    #We use the Hautus correction to avoid zero cells
    nR_S1 = nR_S1 + 0.5
    nR_S2 = nR_S2 + 0.5
     
    hit_count = np.sum(nR_S2[4:])
    miss_count = np.sum(nR_S2[:4])
    fa_count = np.sum(nR_S1[4:])
    cr_count = np.sum(nR_S1[:4])
    
    hit_rate = hit_count / (hit_count + miss_count)
    fa_rate = fa_count / (fa_count + cr_count)
 
    c1 = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))
    # Meta-d prime calculations via Maximum Likelihood Estimation

    trial_df = pd.DataFrame({
        "Stimuli": np.array(actually_correct_list, dtype = int),
        "Responses": (np.array(ratings_eight) > 4).astype(int),
        "Confidence": np.where(
            np.array(ratings_eight) > 4,
            np.array(ratings_eight) - 4,
            5 - np.array(ratings_eight)
        )
    })

    fit = metad(
        data = trial_df, nRatings = 4, stimuli = "Stimuli", confidence = "Confidence", verbose = 0
    )
    meta_d = fit["meta_d"].values[0]
    d1 = fit["dprime"].values[0]
    M_ratio = fit["m_ratio"].values[0]

    auroc2 = roc_auc_score(
        np.array(actually_correct_list, dtype = int),
        np.array(ratings_eight)
    )

    return {
        "d_prime": round(d1, 3),
        "M_ratio": round(M_ratio, 3),
        "AUROC2": round(auroc2, 3),
        "n_items": len(actually_correct_list)
    }