import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from sklearn.preprocessing import scale
import pickle
import tensorflow as tf
import os
from pathlib import Path
from random import sample
import random
from sklearn.model_selection import KFold

def simulate_data(N=100000, # sample size
    seqlen=200,			    # length of sequences
    h2=0.80,				# variability explained by sequence
    p=10,					# number of motifs
    q=4, 					# number of interactions
    alpha=0.4, 				# proportion of motif h2 due to simple additive effects
    interactgap=3,			# max gap size between local interactions
    pi=0.6,					# proportion of p motifs to have non-zero additive effects
    delta = 1.5, 			# motif-positional effect weight
    Iw=np.array([.25,.25,	# how to weight the (1-alpha)* h2 amongst interactions
    			 .25,.25]), # # must sum to 1
    psi=0.3,				# fraction of h2 explained by sequence effects
    curargument=0):

    # Load our preselected motifs,
    # establish lengths (motlens) as a vector of length p.
    pfmsuse=load_motifs()
    motlens = [x.shape[0] for x in pfmsuse]

    # Generate motif frequencies: sample 5 values from Uniform(0.1, 0.4) and
    #  5 from Uniform(0.4, .8)
    motfreqs = np.concatenate((np.random.uniform(0.1, 0.4, 5),
                                np.random.uniform(0.4, 0.8, 5)))
    np.random.shuffle(motfreqs)  

    # Generate a binary matrix M (N x p) where each column i is drawn from Bernoulli(motfreqs[i])
    M = np.empty((N, p), dtype=int)
    for i in range(p):
        M[:, i] = np.random.binomial(1, motfreqs[i], size=N)

    # 0-indexed motif names: "motif0", "motif1", ..., "motif9"
    motif_names = ["motif{}".format(i) for i in range(p)]

    # Generate Mpos: each row corresponds to the positions of the 10 motifs in one sequence.
    Mpos, mu, sigma = generate_all_positions(N, seqlen, motlens)
    motpos_names = ["motif{}".format(i) for i in range(p)]

    # First, get all the sequences and potentially inject some
    # splice-like effects. 
    ## embed_motifs(sequence, motif_positions, motif_seqs):
    sequences = [generate_random_sequence(seqlen) for _ in range(N)]
    print("Realizing motifs")
    real_mots = np.array(realize_motifs(pfmsuse, N=N,motlens=motlens)).T
    inputstrings= [embed_motifs(sequence=sequences[i],
    	motif_positions=list(Mpos[i]),
    	motif_seqs=list(real_mots[i]),
    	to_embed=list(M[i])) for i in range(N)]

    raw_seq_effect=np.array([combined_sequence_score(x) for x in inputstrings])
    # Standardize the sequence effect and scale to have variance psi * h2.
    sequence_effect = scale(raw_seq_effect) * np.sqrt(psi * h2)


    	# --- Define Interaction Pairs ---
    # Create all combinations (0-indexed) of motif indices
    allcombs = list(combinations(np.arange(0, p), 2))

    # Choose 3 distinct interaction pairs randomly
    interactpairs = np.random.choice(np.arange(len(allcombs)), size=3, replace=False)
    simplemotinteracts = allcombs[interactpairs[0]]
    upstreaminteracts = allcombs[interactpairs[1]]
    distineracts = allcombs[interactpairs[2]]

    # For upstream interactions: indicator = 1 if position of first motif < position of second motif.
    upstreamindic = (Mpos[:, upstreaminteracts[0]] < Mpos[:, upstreaminteracts[1]]).astype(int)

    # For distance interactions: indicator = 1 if abs(pos1 - pos2) <= motlens (for that pair) + interactgap.
    distindic = np.abs(Mpos[:, distineracts[0]] - Mpos[:, distineracts[1]])
    distindic = (distindic <= (motlens[distineracts[0]] + interactgap)).astype(int)
    # Note: We use the length of the first motif in the pair; you could choose other logic.

    # For high-order interactions: sample 3 distinct motifs (0-indexed) such that none of the chosen pairs are fully contained.
    highordinteracts = np.random.choice(np.arange(0, p), size=3, replace=False)
    def all_in(a, b):
        return all(x in b for x in a)

    pairexists = (all_in(simplemotinteracts, highordinteracts) or 
                  all_in(upstreaminteracts, highordinteracts) or 
                  all_in(distineracts, highordinteracts))
    while pairexists:
        highordinteracts = np.random.choice(np.arange(0, p), size=3, replace=False)
        pairexists = (all_in(simplemotinteracts, highordinteracts) or 
                      all_in(upstreaminteracts, highordinteracts) or 
                      all_in(distineracts, highordinteracts))

    	# --- Construct Interaction Variables I ---
    # simple: product of the two motif indicators for simplemotinteracts.
    simple = M[:, simplemotinteracts[0]] * M[:, simplemotinteracts[1]]
    # upstream: product of motif indicators for upstream pair, times the upstream indicator.
    upstream = upstreamindic * (M[:, upstreaminteracts[0]] * M[:, upstreaminteracts[1]])
    # distance: product of motif indicators for distance pair, times the distance indicator.
    # distance = distindic * (M[:, distineracts[0]] * M[:, distineracts[1]])
    # high-order interactions: product of the three motif indicators for highordinteracts.
    highord = (M[:, highordinteracts[0]] *
               M[:, highordinteracts[1]] *
               M[:, highordinteracts[2]])
    # Stack interaction features into a matrix I with 4 columns.
    I = np.column_stack([simple, upstreamindic, distindic, highord])
    I_names = ["simple_{s1}_{s2}".format(s1=simplemotinteracts[0],s2=simplemotinteracts[1]), 
                "upstream_{u1}_{u2}".format(u1=upstreaminteracts[0],u2=upstreaminteracts[1]), 
                "distance_{d1}_{d2}".format(d1=distineracts[0],d2=distineracts[1]), 
                "highordinteracts_{ho1}_{ho2}_{ho3}".format(ho1=highordinteracts[0],
                                                            ho2=highordinteracts[1],
                                                            ho3=highordinteracts[2])]

    	# --- Simulate Outcome (Liability) ---
    # Generate effect sizes:
    # Standardize M and I using sklearn's scale function.
    M_scaled = scale(M)
    I_scaled = scale(I)

    beta=np.zeros(p)
    numactive=int(np.round(pi*p))
    # note when delta is 0 there's no position weighting
    sigma_beta = np.sqrt((alpha * (1 - psi) * h2) / (numactive * (1 + delta**2 / 3)))
    idxactive=np.random.choice(np.arange(p), size=numactive, replace=False)
    beta[idxactive] = np.random.normal(loc=0, scale=sigma_beta, size=numactive)
    effective_effects = np.empty((N, p))

    # Initialize mu and sigma for positional preferences
    mu = np.zeros(p)    # preferred positions
    sigma = np.ones(p)  # spread of positional effect
    motifs_with_decay = np.random.choice(p, size=5, replace=False)
    for i in range(p):
        if i in motifs_with_decay:
            # Sample preferred position from within sequence range
            mu[i] = np.random.randint(0, seqlen)
            # Sample spread for positional preference
            sigma[i] = np.random.uniform(10, 30)
        else:
            mu[i] = 0
            sigma[i] = np.inf  # Effectively disables decay

    # Compute decay-modified effects
    for i in range(p):
        # f_i is 1.0 when sigma is very large (i.e., motif has no preference)
        f_i = np.exp(- ((Mpos[:, i] - mu[i]) ** 2) / (2 * sigma[i] ** 2))
        # Apply delta to control the strength of position effect
        weighted_f_i = (1 - delta) + delta * f_i
        effective_effects[:, i] = M_scaled[:, i] * beta[i] * weighted_f_i


    # The overall motif contribution for each sequence:
    motif_add_effect = effective_effects.sum(axis=1)

    # Allow interactions' importance be specified by Iw
    # Adjust interactive effect sizes so that interactions explain ((1 - alpha) * (1 - psi) * h2).
    gamma = np.array([np.random.normal(loc=0, scale=np.sqrt(x * (1 - alpha) * (1 - psi) * h2))
                      for x in Iw])
    motif_int_effect = np.dot(I_scaled, gamma)

    eps = np.random.normal(loc=0, scale=np.sqrt(1 - h2), size=N)
    # Compute liability: linear combination of scaled M and I plus noise.
    liabs = motif_add_effect + motif_int_effect + sequence_effect + eps

    savedir="/u/project/halperin/mjthomps/motif_sims/interim_files/"
    savesuffix="_h2_{h2}_p_{p}_q_{q}_alpha_{alpha}_N_{N}_".format(
            h2=h2, p=p, q=q, alpha=alpha, N=N)+\
            "pi_{pi}_delta_{delta}_psi_{psi}_run_{j}.txt".format(
            pi=pi,delta=delta,psi=psi,j=curargument)

    df_explain = pd.DataFrame(np.hstack([M, I, liabs.reshape(-1, 1)]),
                          columns=motif_names + I_names + ["liab"])
    df_pos = pd.DataFrame(Mpos, columns=motpos_names)
    # Create an effect sizes DataFrame.
    M_means = np.mean(M, axis=0)
    I_means = np.mean(I, axis=0)

    newnames=[]
    for i in range(p):
        wdecay=i in motifs_with_decay
        newnames.append(motif_names[i]+"_"+str(wdecay)+"_"+str(mu[i]))

    eff_varnames = newnames + I_names
    eff_sizes = np.concatenate([beta, gamma])
    freqs = np.concatenate([M_means, I_means])
    effdf = pd.DataFrame({
        "varname": eff_varnames,
        "effsize": eff_sizes,
        "freqs": freqs
    })
    df_explain.to_csv(savedir+"explaindf"+savesuffix,sep="\t")
    df_pos.to_csv(savedir+"posdf"+savesuffix,sep="\t")
    effdf.to_csv(savedir+"effdf"+savesuffix,sep="\t")
    print("Saved information dataframes, dumping pickle")
    nndir="/u/project/halperin/mjthomps/motif_sims/nn_data/data"+savesuffix
    Xtr, Ytr, Xte, Yte=get_OH_train_test(inputstrings, liabs,nndir)

    # now return this data for training models
    return Xtr, Ytr, Xte, Yte, savesuffix

def get_save_suffix(N=100000, # sample size
    seqlen=200,             # length of sequences
    h2=0.80,                # variability explained by sequence
    p=10,                   # number of motifs
    q=4,                    # number of interactions
    alpha=0.4,              # proportion of motif h2 due to simple additive effects
    interactgap=3,          # max gap size between local interactions
    pi=0.6,                 # proportion of p motifs to have non-zero additive effects
    delta = 1.5,            # motif-positional effect weight
    Iw=np.array([.25,.25,   # how to weight the (1-alpha)* h2 amongst interactions
                 .25,.25]), # # must sum to 1
    psi=0.3,                # fraction of h2 explained by sequence effects
    curargument=0):
    savesuffix="_h2_{h2}_p_{p}_q_{q}_alpha_{alpha}_N_{N}_".format(
            h2=h2, p=p, q=q, alpha=alpha, N=N)+\
            "pi_{pi}_delta_{delta}_psi_{psi}_run_{j}.txt".format(
            pi=pi,delta=delta,psi=psi,j=curargument)
    return savesuffix

def generate_random_sequence(length):
    return ''.join(np.random.choice(list('ACGT'), size=length))

def embed_motifs(sequence, motif_positions, motif_seqs,to_embed):
    seq_list = list(sequence)
    for pos, motif,te in zip(motif_positions, motif_seqs,to_embed):
    	if te:
        	seq_list[pos:pos+len(motif)] = list(motif)
    return ''.join(seq_list)

# Function to check if any two motifs (with their lengths) overlap.
def generate_all_positions(N, seqlen, motlens):
    """
    Generate a matrix of motif starting positions for N sequences without overlaps.
    Positions are 0-indexed. 
      - Motif0 is forced to lie within the first 5 positions.
      - Motif(p-1) is forced to lie within the last 5 valid positions.
      - Other motifs are sampled uniformly over their allowed ranges.
    Uses vectorized overlap checking and resamples only for sequences with overlaps.
    
    Parameters:
      N: number of sequences.
      seqlen: length of the sequence.
      motlens: array-like of motif lengths (length p).
    
    Returns:
      positions: an array of shape (N, p) of non-overlapping starting positions.
    """
    motlens = np.array(motlens)
    p = len(motlens)
    
    # Pre-compute allowed ranges for each motif.
    allowed_ranges = []
    mu = np.empty(p)
    sigma = np.empty(p)
    # for i in range(p):
    #     if i == 0:
    #         # Motif0: allowed positions from 0 to min(5, seqlen - motlens[0] + 1)
    #         max_valid = seqlen - motlens[i] + 1
    #         upper_bound = min(5, max_valid)
    #         allowed_ranges.append(np.arange(0, upper_bound))
    #         rng = np.arange(0, upper_bound)
    #         mu[i] = (rng[0] + rng[-1]) / 2
    #         sigma[i] = (rng[-1] - rng[0]) / 2 if (rng[-1]-rng[0]) > 0 else 1
    #     elif i == p - 1:
    #         # Motif(p-1): allowed positions from max(0, seqlen - motlens[i] - 5) to seqlen - motlens[i] + 1
    #         min_valid = seqlen - motlens[i] - 5
    #         if min_valid < 0:
    #             min_valid = 0
    #         allowed_ranges.append(np.arange(min_valid, seqlen - motlens[i] + 1))
    #         rng = np.arange(min_valid, seqlen - motlens[i] + 1)
    #         mu[i] = (rng[0] + rng[-1]) / 2
    #         sigma[i] = (rng[-1] - rng[0]) / 2 if (rng[-1]-rng[0]) > 0 else 1
    #     elif i < 5 and i !=0:
    #         # First half motifs stronger the closer to center they are
    #         max_valid = int(seqlen/2)
    #         upper_bound = max(5, max_valid)
    #         allowed_ranges.append(np.arange(5, upper_bound))
    #         rng = np.arange(5, upper_bound)
    #         mu[i] = (rng[0] + rng[-1]) / 2
    #         sigma[i] = (rng[-1] - rng[0]) / 2 if (rng[-1]-rng[0]) > 0 else 1
    #     else:
    #         # Second half motifs stronger closer to center they are
    #         min_valid = int(seqlen/2)
    #         if min_valid < 0:
    #             min_valid = 0
    #         allowed_ranges.append(np.arange(min_valid, seqlen - motlens[i] - 5))
    #         rng = np.arange(min_valid, seqlen - motlens[i] + 1)
    #         mu[i] = (rng[0] + rng[-1]) / 2
    #         sigma[i] = (rng[-1] - rng[0]) / 2 if (rng[-1]-rng[0]) > 0 else 1
    
    # Just uniformly disperse them throughout the construct
    for i in range(p):
        allowed_ranges.append(np.arange(0, seqlen - motlens[i]))

    # Sample initial candidate positions for all N sequences.
    positions = np.empty((N, p), dtype=int)
    for i in range(p):
        rng = allowed_ranges[i]
        positions[:, i] = np.random.choice(rng, size=N, replace=True)
    
    # Vectorized overlap check.
    def check_overlap(pos_array):
        # pos_array shape: (n, p)
        A = pos_array[:, :, None]    # shape (n, p, 1)
        B = pos_array[:, None, :]    # shape (n, 1, p)
        L_i = motlens.reshape(1, p, 1)  # shape (1, p, 1)
        L_j = motlens.reshape(1, 1, p)  # shape (1, 1, p)
        # Two intervals overlap if: A < B + L_j and B < A + L_i
        overlap = (A < (B + L_j)) & (B < (A + L_i))
        # Create a mask for the upper triangle to ignore self-comparison
        mask = np.triu(np.ones((p, p), dtype=bool), k=1)
        # For each sequence, return True if any pair (in the upper triangle) overlaps.
        return np.any(overlap[:, mask], axis=1)
    
    # Check overlaps for all sequences.
    valid_seq = ~check_overlap(positions)
    invalid_indices = np.where(~valid_seq)[0]
    
    # Use a progress bar to monitor resampling of invalid sequences.
    pbar = tqdm(total=len(invalid_indices), desc="Resampling invalid sequences")
    iteration = 0
    while len(invalid_indices) > 0:
        iteration += 1
        # Resample positions for only the invalid sequences.
        for i in range(p):
            rng = allowed_ranges[i]
            positions[invalid_indices, i] = np.random.choice(rng, size=len(invalid_indices), replace=True)
        # Re-check overlaps for these sequences.
        still_invalid = check_overlap(positions[invalid_indices])
        num_invalid = np.sum(still_invalid)
        pbar.update(len(invalid_indices) - num_invalid)
        # Update invalid indices.
        invalid_indices = invalid_indices[still_invalid]
    pbar.close()
    return positions, mu, sigma

def load_motifs():
    with open("/u/project/halperin/mjthomps/motif_sims/notebooks/pfms.pickle","rb") as f:
    	pfmsuse=pickle.load(f)
    return pfmsuse

def realize_motifs(pfmsuse=None,N=100000,motlens=None):
    motstoinsert=[]
    for p in range(len(pfmsuse)):
        lets=[]
        for k in range(motlens[p]):
            lets.append(np.random.choice(["A","C","T","G"],N,p=pfmsuse[p][k,:]))
        lets=np.array(lets).transpose()
        motstoinsert.append(list([''.join(row) for row in lets]))
    return motstoinsert

nts="ACGT"

def str_to_vector(str, template):
    #   return [ei_vec(template.index(nt),len(template)) for nt in str]
    mapping = dict(zip(template, range(len(template))))
    seq = [mapping[i] for i in str]
    return np.eye(len(template))[seq]

def seq_to_vector(seq):
    return str_to_vector(seq, nts)

from sklearn.model_selection import KFold
def get_OH_train_test(stringsIN,labels,savesuffix):
    kf=KFold(n_splits=10,shuffle=True,random_state=1999)


    ### now make actual nn inputs
    in_seqs=[seq_to_vector(x) for x in stringsIN]
    in_seqs=np.array(in_seqs)
    instrings=np.array(stringsIN)
    fitness=labels

    folds=list(kf.split(labels))
    testidx=folds[9][1]

    #     instrings=df[seqpart].values
    Xtr=[np.delete(in_seqs,testidx,axis=0),
        np.delete(instrings,testidx,axis=0)]
    Xte=[in_seqs[testidx,:,:],
        instrings[testidx]]

    Ytr=np.delete(fitness,testidx).astype(float)
    Yte=fitness[testidx].astype(float)

    shuffleidx_train=sample(range(Xtr[0].shape[0]),Xtr[0].shape[0])
    Xtr=[Xtr[0][shuffleidx_train,:,:],
        Xtr[1][shuffleidx_train]]
    Ytr=Ytr[shuffleidx_train]
    shuffleidx_test=sample(range(Xte[0].shape[0]),Xte[0].shape[0])
    Xte=[Xte[0][shuffleidx_test,:,:],
        Xte[1][shuffleidx_test]]
    Yte=Yte[shuffleidx_test]
    with open(savesuffix+".pickle","wb") as f:
        pickle.dump([Xtr,Ytr,Xte,Yte], 
                    f) 
    return [Xtr, Ytr, Xte, Yte]


def left_to_right_trajectory(seq, threshold=1.2, w_pos=0.06, w_neg=-0.02, 
                               refractory_steps=5, leak=0.99, 
                               adaptation_increase=0.3, adaptation_decay=0.95, 
                               start_skip=20, min_interval=10):
    """
    Processes the sequence from left-to-right.
    Rewards 'T' and 'C' with a positive delta.
    Returns two lists of length L:
      - score_traj: running score (cumulative burst + state) at each position.
      - spike_traj: last spike position (index) recorded up to that point.
    """
    L = len(seq)
    state = 0.0
    adaptation = 0.0
    spike_sum = 0.0
    refractory_timer = 0
    last_spike = None
    last_spike_time = -np.inf
    score_traj = [None] * L
    spike_traj = [None] * L

    for i in range(L):
        if i < start_skip:
            score_traj[i] = spike_sum + state
            spike_traj[i] = last_spike
            continue

        if refractory_timer > 0:
            state *= leak
            adaptation *= adaptation_decay
            refractory_timer -= 1
        else:
            # Reward positive delta for letters 'T' and 'C'
            if seq[i] in ['T', 'C']:
                delta = w_pos
            elif seq[i] in ['A', 'G']:
                delta = w_neg
            else:
                delta = 0.0

            state += (delta - adaptation)
            state *= leak

            if state >= threshold:
                # Use quadratic burst for nonlinearity.
                burst = state ** 2
                if i - last_spike_time < min_interval:
                    burst *= 0.5  # penalize if spikes occur too close together
                spike_sum += burst
                last_spike = i
                last_spike_time = i
                adaptation += adaptation_increase
                refractory_timer = refractory_steps
                state = -1.0  # strong reset
            adaptation *= adaptation_decay

        score_traj[i] = spike_sum + state
        spike_traj[i] = last_spike

    return score_traj, spike_traj

def right_to_left_trajectory_with_pattern(seq, threshold=1.2, w_pos=0.6, w_neg=-0.02, 
                                          refractory_steps=5, leak=0.99, 
                                          adaptation_increase=0.3, adaptation_decay=0.95, 
                                          start_skip=20, min_interval=10, pattern="GA"):
    """
    Processes the sequence from right-to-left.
    Rewards the two-character pattern (default "GA") in the reversed sequence.
    Returns two lists (length L):
      - score_traj: running score (cumulative burst + state) at each position (original indexing).
      - spike_traj: last spike position (original index) recorded up to that point.
    """
    L = len(seq)
    state = 0.0
    adaptation = 0.0
    spike_sum = 0.0
    refractory_timer = 0
    last_spike = None
    last_spike_time = -np.inf
    score_traj = [None] * L
    spike_traj = [None] * L

    steps_processed = 0
    # Process from right-to-left: indices L-1 down to 0.
    for i in range(L-1, -1, -1):
        steps_processed += 1
        if steps_processed < start_skip:
            score_traj[i] = spike_sum + state
            spike_traj[i] = last_spike
            continue

        if refractory_timer > 0:
            state *= leak
            adaptation *= adaptation_decay
            refractory_timer -= 1
        else:
            # Check for two-character pattern in right-to-left order.
            # For position i, look at seq[i] and seq[i-1] if available.
            if i - 1 >= 0 and (seq[i] + seq[i-1]) == pattern:
                delta = w_pos
            else:
                delta = w_neg

            state += (delta - adaptation)
            state *= leak

            if state >= threshold:
                burst = state ** 2
                if steps_processed - last_spike_time < min_interval:
                    burst *= 0.5
                spike_sum += burst
                last_spike = i  # i is in original indexing.
                last_spike_time = steps_processed
                adaptation += adaptation_increase
                refractory_timer = refractory_steps
                state = -1.0
            adaptation *= adaptation_decay

        score_traj[i] = spike_sum + state
        spike_traj[i] = last_spike

    return score_traj, spike_traj

# Bonus functions remain the same.
def bonus_lr(spike_pos, L):
    """Return bonus of 1 if left-to-right spike is between 170 and L (e.g., 200), else 0."""
    if spike_pos is None:
        return 0.0
    return 1.0 if 170 <= spike_pos <= L else 0.0

def bonus_rl(spike_pos):
    """Return bonus of 1 if right-to-left spike is between 1 and 20, else 0."""
    if spike_pos is None:
        return 0.0
    return 1.0 if 1 <= spike_pos <= 20 else 0.0

def combined_trajectory(seq, L=None, bonus_factor_lr=0.5, bonus_factor_rl=0.5, penalty_coef=0.001):
    """
    Computes the per-position trajectories for:
      - Left-to-right recurrence (driven by 'T' and 'C')
      - Right-to-left recurrence (driven by the "GA" pattern)
    Then, at the final positions we combine the two trajectories.
    
    We add a bonus if:
      - The left-to-right spike occurs between 170 and L.
      - The right-to-left spike occurs between 1 and 20.
    
    We apply a linear penalty if the gap between these spikes is smaller than desired.
    
    Returns:
      - lr_score, lr_spike: trajectories from left-to-right.
      - rl_score, rl_spike: trajectories from right-to-left.
      - combined_traj: elementwise combined trajectory (tanh(lr_score + rl_score)).
      - combined_final: final combined score computed from the final spike positions.
    """
    if L is None:
        L = len(seq)
    lr_score, lr_spike = left_to_right_trajectory(seq, threshold=1.2, w_pos=0.06, w_neg=-0.02,
                                                  refractory_steps=5, leak=0.99,
                                                  adaptation_increase=0.3, adaptation_decay=0.95,
                                                  start_skip=20, min_interval=10)
    rl_score, rl_spike = right_to_left_trajectory_with_pattern(seq, threshold=1.2, w_pos=0.6, w_neg=-0.02,
                                                               refractory_steps=5, leak=0.99,
                                                               adaptation_increase=0.3, adaptation_decay=0.95,
                                                               start_skip=20, min_interval=10, pattern="GA")
    # Compute bonuses from the final spike positions.
    bonus_from_lr = bonus_factor_lr * bonus_lr(lr_spike[-1], L)
    bonus_from_rl = bonus_factor_rl * bonus_rl(rl_spike[0])
    # Compute gap: left-to-right spike (at end) minus right-to-left spike (at beginning).
    if lr_spike[-1] is not None and rl_spike[0] is not None:
        gap = lr_spike[-1] - rl_spike[0]
    else:
        gap = 0
    # Desired gap: ideally, for a 200-base sequence, a gap of 160 (i.e. spikes in positions ~200 and ~40).
    desired_gap = L - 40
    # Use a linear penalty (much smaller than before).
    penalty = penalty_coef * max(0, (desired_gap - gap))
    
    # Final combined score: use the final scores from each recurrence plus bonuses minus penalty.
    combined_final = np.tanh(lr_score[-1] + rl_score[0] + bonus_from_lr + bonus_from_rl - penalty)
    
    # Also compute an elementwise combined trajectory.
    combined_traj = [np.tanh(lr_score[i] + rl_score[i]) for i in range(L)]
    
    return lr_score, lr_spike, rl_score, rl_spike, combined_traj, combined_final

def combined_sequence_score(seq):
    x,y,z,a,b, combined_final=combined_trajectory(seq)
    return combined_final






