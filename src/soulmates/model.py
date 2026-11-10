"""PyMC model for new contacts per day by age band and country, and the chance of finding your soul mate."""
import numpy as np
import preliz as pz
import pymc as pm
import pytensor.tensor as pt

N_BANDS = 17  # 5 year bands, the last one holds ages 80 and over


def age_band(age):
    return np.minimum(np.asarray(age) // 5, N_BANDS - 1)


def recog_prior(bounds=(0.3, 0.9), mass=0.9):
    # The speaker's guess, "I am mass sure p_recog lies between the bounds", turned into the
    # flattest Beta prior that meets it (maximum entropy), so the answer is unique
    recog = pz.Beta()
    pz.maxent(recog, lower=bounds[0], upper=bounds[1], mass=mass, plot=False)
    return float(recog.alpha), float(recog.beta)


def build_model(df, search_ages=(15, 60), recog_bounds=(0.3, 0.9), recog_mass=0.9):
    band = age_band(df["age"].to_numpy())
    weekend = df["weekend"].to_numpy().astype(float)
    countries, country = np.unique(df["country"].to_numpy().astype(str), return_inverse=True)
    search = np.arange(search_ages[0], search_ages[1] + 1)
    years_per_band = np.bincount(age_band(search), minlength=N_BANDS)  # search years that fall in each band
    with pm.Model(coords={"band": np.arange(N_BANDS), "country": countries}) as model:
        mu_pop = pm.Normal("mu_pop", np.log(0.4), 1)  # guess: about 0.4 new people of your age per day
        sigma_band = pm.HalfNormal("sigma_band", 0.5)
        z = pm.Normal("z", 0, 1, dims="band")  # non centred form
        band_mean = pm.Deterministic("band_mean", pt.exp(mu_pop + sigma_band * z), dims="band")  # typical country
        sigma_country = pm.HalfNormal("sigma_country", 0.5)  # how far countries sit from the typical one
        # The country terms sum to zero, so mu_pop is the average of the observed countries. Without
        # that, the overall level and the country levels can trade off and the sampler crawls.
        log_factor = pm.ZeroSumNormal("log_factor", sigma=sigma_country, dims="country")
        country_factor = pm.Deterministic("country_factor", pt.exp(log_factor), dims="country")
        beta_weekend = pm.Normal("beta_weekend", 0, 0.5)  # log of the weekend multiplier, centred on no effect
        alpha = pm.HalfNormal("alpha", 5)
        pm.NegativeBinomial(
            "new_contacts",
            mu=band_mean[band] * country_factor[country] * pt.exp(beta_weekend * weekend),
            alpha=alpha,
            observed=df["new_contacts"].to_numpy(),
        )
        week = 5 / 7 + 2 / 7 * pt.exp(beta_weekend)  # average day of a week with 5 weekdays and 2 weekend days
        typical = pm.Deterministic("lifetime_typical", 365 * week * pt.sum(years_per_band * band_mean))
        pm.Deterministic("lifetime_country", typical * country_factor, dims="country")
        # a country we have no data for: a new draw from the distribution the observed countries came from
        z_new = pm.Normal("z_new", 0, 1)
        pm.Deterministic("lifetime_anywhere", typical * pt.exp(sigma_country * z_new))
        a, b = recog_prior(recog_bounds, recog_mass)
        pm.Beta("p_recog", alpha=a, beta=b)
    return model


def p_find(idata, pool_size, lifetime="lifetime_anywhere"):
    post = idata.posterior
    # 1 − (1 − p_recog / M) ^ lifetime, written so a tiny probability does not round to zero
    return -np.expm1(post[lifetime] * np.log1p(-post["p_recog"] / pool_size))
