import numpy as np
import bct


def threshold_adj(A, q=0.8, abs=True, fill_diag=True, binarize=True):
    if abs:
        A = np.abs(A)

    if fill_diag:
        np.fill_diagonal(A, 0)

    thresh = np.quantile(A, q=q)
    mask = A >= thresh

    A_out = A.copy()
    A_out[~mask] = 0
    if binarize:
        A_out[mask] = 1

    return A_out


def get_norm_rc(A, n_perms=1000, directed=True, weighted=True):
    if directed == True:
        _, _, degree = bct.degrees_dir(A)
        kmax = int(np.max(degree))
        if weighted == True:
            R = bct.rich_club_wd(A, klevel=kmax)
        elif weighted == False:
            R, _, _ = bct.rich_club_bd(A, klevel=kmax)

        if n_perms > 0:
            R_perm = np.zeros((n_perms, kmax))
            for i in np.arange(n_perms):
                np.random.seed(i)
                A_rand, _ = bct.randmio_dir(A, itr=5)
                if weighted == True:
                    R_perm[i, :] = bct.rich_club_wd(A_rand, klevel=kmax)
                elif weighted == False:
                    R_perm[i, :], _, _ = bct.rich_club_bd(A_rand, klevel=kmax)
    elif directed == False:
        pass  # linden, fix this

    # compute normalized rich club coefficient
    if n_perms > 0:
        R_norm = np.divide(R, np.nanmean(R_perm, axis=0))

        # compute p values
        p_val = np.zeros(kmax)
        for i in np.arange(kmax):
            p_val[i] = np.nanmean(R[i] <= R_perm[:, i])

    if n_perms > 0:
        return R, R_perm, R_norm, p_val
    else:
        return R
