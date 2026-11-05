"""PyMC model for new contacts per day, and the chance of finding your soul mate."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_BANDS = 17  # 5 year bands, the last one holds ages 80 and over


def age_band(age):
    return np.minimum(np.asarray(age) // 5, N_BANDS - 1)


def build_model(df, search_ages=(18, 60)):
    band = age_band(df["age"].to_numpy())
    search = np.arange(search_ages[0], search_ages[1] + 1)
    years_per_band = np.bincount(age_band(search), minlength=N_BANDS)  # search years that fall in each band
    with pm.Model(coords={"band": np.arange(N_BANDS)}) as model:
        mu_pop = pm.Normal("mu_pop", np.log(2), 1)  # guess: about 2 new people per day, wide on purpose
        sigma_band = pm.HalfNormal("sigma_band", 0.5)
        z = pm.Normal("z", 0, 1, dims="band")  # non centred form
        band_mean = pm.Deterministic("band_mean", pt.exp(mu_pop + sigma_band * z), dims="band")
        alpha = pm.HalfNormal("alpha", 5)
        pm.NegativeBinomial(
            "new_contacts", mu=band_mean[band], alpha=alpha, observed=df["new_contacts"].to_numpy()
        )
        lifetime_new = pm.Deterministic("lifetime_new", 365 * pt.sum(years_per_band * band_mean))
        p_eye = pm.Beta("p_eye", 2, 2)  # guess, no data informs it
        pm.Deterministic("eye_contacts", lifetime_new * p_eye)
        pm.Beta("p_recog", 2, 2)  # guess, no data informs it
    return model


def p_find(idata, pool_size):
    post = idata.posterior
    # 1 − (1 − p_recog / M) ^ eye_contacts, written so a tiny probability does not round to zero
    return -np.expm1(post["eye_contacts"] * np.log1p(-post["p_recog"] / pool_size))
