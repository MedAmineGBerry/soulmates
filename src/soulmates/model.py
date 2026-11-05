"""PyMC model for new contacts per day, and the chance of finding your soul mate."""
import numpy as np
import preliz as pz
import pymc as pm
import pytensor.tensor as pt

N_BANDS = 17  # 5 year bands, the last one holds ages 80 and over


def age_band(age):
    return np.minimum(np.asarray(age) // 5, N_BANDS - 1)


def build_model(df, search_ages=(18, 60), recog_bounds=(0.1, 0.6), recog_mass=0.9):
    band = age_band(df["age"].to_numpy())
    weekend = df["weekend"].to_numpy().astype(float)
    search = np.arange(search_ages[0], search_ages[1] + 1)
    years_per_band = np.bincount(age_band(search), minlength=N_BANDS)  # search years that fall in each band
    with pm.Model(coords={"band": np.arange(N_BANDS)}) as model:
        mu_pop = pm.Normal("mu_pop", np.log(2), 1)  # guess: about 2 new people per day, wide on purpose
        sigma_band = pm.HalfNormal("sigma_band", 0.5)
        z = pm.Normal("z", 0, 1, dims="band")  # non centred form
        band_mean = pm.Deterministic("band_mean", pt.exp(mu_pop + sigma_band * z), dims="band")
        beta_weekend = pm.Normal("beta_weekend", 0, 0.5)  # log of the weekend multiplier, centred on no effect
        alpha = pm.HalfNormal("alpha", 5)
        pm.NegativeBinomial(
            "new_contacts",
            mu=band_mean[band] * pt.exp(beta_weekend * weekend),
            alpha=alpha,
            observed=df["new_contacts"].to_numpy(),
        )
        week = 5 / 7 + 2 / 7 * pt.exp(beta_weekend)  # average day of a week with 5 weekdays and 2 weekend days
        pm.Deterministic("lifetime_new", 365 * week * pt.sum(years_per_band * band_mean))
        # The speaker's guess, "I am recog_mass sure p_recog lies between the bounds", turned into the
        # flattest Beta prior that meets it (maximum entropy), so the answer is unique
        recog = pz.Beta()
        pz.maxent(recog, lower=recog_bounds[0], upper=recog_bounds[1], mass=recog_mass, plot=False)
        pm.Beta("p_recog", alpha=float(recog.alpha), beta=float(recog.beta))
    return model


def p_find(idata, pool_size):
    post = idata.posterior
    # 1 − (1 − p_recog / M) ^ lifetime_new, written so a tiny probability does not round to zero
    return -np.expm1(post["lifetime_new"] * np.log1p(-post["p_recog"] / pool_size))
