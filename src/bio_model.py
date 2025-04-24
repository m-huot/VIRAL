import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd


def bio_model(pheno):
    pheno = np.copy(pheno)

    pheno[:, 0] = pheno[:, 0] - 9.0
    pheno[:, 1] = pheno[:, 1] - 9.8
    pheno[:, 2] = pheno[:, 2] - 10.2
    pheno[:, 3] = pheno[:, 3] - 10.0
    pheno[:, 4] = pheno[:, 4] - 9.3

    T = 1.6
    a = 1.57363338
    C = 5.47640269e-07
    K_1 = 5.60154369e-08
    K_2 = 4.51280184e-08
    K_3 = 7.18248667e-08
    K_4 = 4.72731727e-07

    return (
        a
        * (1 + C * np.exp(-np.log(10) * pheno[:, 0] / T))
        / (
            C * np.exp(-np.log(10) * pheno[:, 0] / T)
            + K_1 * np.exp(-np.log(10) * pheno[:, 1] / T)
            + K_2 * np.exp(-np.log(10) * pheno[:, 2] / T)
            + K_3 * np.exp(-np.log(10) * pheno[:, 3] / T)
            + K_4 * np.exp(-np.log(10) * pheno[:, 4] / T)
            + 1
        )
    )


def bio_model_var(var, pheno):
    """
    variance in fitness function: V(Kd) -> V(fitness)
    """
    pheno = np.copy(pheno)
    pheno[:, 0] = pheno[:, 0] - 9.0
    pheno[:, 1] = pheno[:, 1] - 9.8
    pheno[:, 2] = pheno[:, 2] - 10.2
    pheno[:, 3] = pheno[:, 3] - 10.0
    pheno[:, 4] = pheno[:, 4] - 9.3

    # DERIVATIVES:
    log10Kd_ACE2, log10Kd_CB6, log10Kd_CoV555, log10Kd_REGN10987, log10Kd_S309 = (
        sp.symbols(
            "log10Kd_ACE2 log10Kd_CB6 log10Kd_CoV555 log10Kd_REGN10987 log10Kd_S309"
        )
    )

    T = 1.6
    a = 1.57363338
    C = 5.47640269e-07
    K_1 = 5.60154369e-08
    K_2 = 4.51280184e-08
    K_3 = 7.18248667e-08
    K_4 = 4.72731727e-07

    # Define the bio model function
    def bio_model_bis(
        log10Kd_ACE2, log10Kd_CB6, log10Kd_CoV555, log10Kd_REGN10987, log10Kd_S309
    ):
        return (
            a
            * (1 + C * sp.exp(-sp.log(10) * log10Kd_ACE2 / T))
            / (
                C * sp.exp(-sp.log(10) * log10Kd_ACE2 / T)
                + K_1 * sp.exp(-sp.log(10) * log10Kd_CB6 / T)
                + K_2 * sp.exp(-sp.log(10) * log10Kd_CoV555 / T)
                + K_3 * sp.exp(-sp.log(10) * log10Kd_REGN10987 / T)
                + K_4 * sp.exp(-sp.log(10) * log10Kd_S309 / T)
                + 1
            )
        )

    # Compute the partial derivatives
    df_dlog10Kd_ACE2 = sp.diff(
        bio_model_bis(
            log10Kd_ACE2, log10Kd_CB6, log10Kd_CoV555, log10Kd_REGN10987, log10Kd_S309
        ),
        log10Kd_ACE2,
    )
    df_dlog10Kd_CB6 = sp.diff(
        bio_model_bis(
            log10Kd_ACE2, log10Kd_CB6, log10Kd_CoV555, log10Kd_REGN10987, log10Kd_S309
        ),
        log10Kd_CB6,
    )
    df_dlog10Kd_CoV555 = sp.diff(
        bio_model_bis(
            log10Kd_ACE2, log10Kd_CB6, log10Kd_CoV555, log10Kd_REGN10987, log10Kd_S309
        ),
        log10Kd_CoV555,
    )
    df_dlog10Kd_REGN10987 = sp.diff(
        bio_model_bis(
            log10Kd_ACE2, log10Kd_CB6, log10Kd_CoV555, log10Kd_REGN10987, log10Kd_S309
        ),
        log10Kd_REGN10987,
    )
    df_dlog10Kd_S309 = sp.diff(
        bio_model_bis(
            log10Kd_ACE2, log10Kd_CB6, log10Kd_CoV555, log10Kd_REGN10987, log10Kd_S309
        ),
        log10Kd_S309,
    )

    var_array = np.zeros(len(var))

    for i in range(len(var)):
        df_dlog10Kd_ACE2 = df_dlog10Kd_ACE2.evalf(
            subs={
                log10Kd_ACE2: pheno[i, 0],
                log10Kd_CB6: pheno[i, 1],
                log10Kd_CoV555: pheno[i, 2],
                log10Kd_REGN10987: pheno[i, 3],
                log10Kd_S309: pheno[i, 4],
            }
        )
        df_dlog10Kd_CB6 = df_dlog10Kd_CB6.evalf(
            subs={
                log10Kd_ACE2: pheno[i, 0],
                log10Kd_CB6: pheno[i, 1],
                log10Kd_CoV555: pheno[i, 2],
                log10Kd_REGN10987: pheno[i, 3],
                log10Kd_S309: pheno[i, 4],
            }
        )
        df_dlog10Kd_CoV555 = df_dlog10Kd_CoV555.evalf(
            subs={
                log10Kd_ACE2: pheno[i, 0],
                log10Kd_CB6: pheno[i, 1],
                log10Kd_CoV555: pheno[i, 2],
                log10Kd_REGN10987: pheno[i, 3],
                log10Kd_S309: pheno[i, 4],
            }
        )
        df_dlog10Kd_REGN10987 = df_dlog10Kd_REGN10987.evalf(
            subs={
                log10Kd_ACE2: pheno[i, 0],
                log10Kd_CB6: pheno[i, 1],
                log10Kd_CoV555: pheno[i, 2],
                log10Kd_REGN10987: pheno[i, 3],
                log10Kd_S309: pheno[i, 4],
            }
        )
        df_dlog10Kd_S309 = df_dlog10Kd_S309.evalf(
            subs={
                log10Kd_ACE2: pheno[i, 0],
                log10Kd_CB6: pheno[i, 1],
                log10Kd_CoV555: pheno[i, 2],
                log10Kd_REGN10987: pheno[i, 3],
                log10Kd_S309: pheno[i, 4],
            }
        )

        derivatives = np.array(
            [
                df_dlog10Kd_ACE2,
                df_dlog10Kd_CB6,
                df_dlog10Kd_CoV555,
                df_dlog10Kd_REGN10987,
                df_dlog10Kd_S309,
            ]
        )
        var_array[i] = np.dot(var[i, :], derivatives**2)

    return var_array


def bio_model_fitness(pheno):
    """
    fitness is the first column of pheno
    """

    pheno = np.copy(pheno)

    return pheno[:, 0]


def bio_model_var_fitness(var, pheno):
    """
    variance in fitness function: V(Kd) -> V(fitness)
    """
    return var[:, 0]
