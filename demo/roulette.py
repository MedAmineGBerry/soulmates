import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from soulmates.roulette import load_inputs, spin

    # written by scripts/export_demo_inputs.py, committed to git so the demo runs from a fresh clone
    inputs = load_inputs(mo.notebook_dir().parent / "data" / "demo_inputs.json")
    return inputs, mo, np, plt, spin


@app.cell
def _(mo):
    mo.md("""
    # Soul Mate Roulette

    Each spin plays out simulated lives. Under soul mate rules, one person in the pool is your match, and you
    have to meet them, talk to them and recognize them. Under real world rules, about 1 in 2,400 of the new
    people you talk to could be a partner, as fitted on the How Couples Meet and Stay Together survey.
    """)
    return


@app.cell
def _(mo):
    age = mo.ui.slider(18, 60, value=30, label="Your age", show_value=True)
    pool = mo.ui.dropdown(
        {"the whole world": "world", "the 8 POLYMOD countries": "europe"},
        value="the whole world",
        label="Your soul mate lives in",
    )
    lower = mo.ui.slider(0.01, 0.99, step=0.01, value=0.1, label="Lower bound for p_recog", show_value=True)
    upper = mo.ui.slider(0.01, 0.99, step=0.01, value=0.6, label="Upper bound for p_recog", show_value=True)
    one = mo.ui.run_button(label="Spin 1 life")
    many = mo.ui.run_button(label="Spin 1,000 lives")
    mo.vstack([mo.hstack([age, pool]), mo.hstack([lower, upper]), mo.hstack([one, many])])
    return age, lower, many, one, pool, upper


@app.cell
def _(lower, mo, upper):
    mo.stop(lower.value >= upper.value, mo.callout("The lower bound has to be below the upper bound.", kind="warn"))
    bounds = (lower.value, upper.value)
    mo.md(f"The guess: I am 90% sure that p_recog lies between {lower.value:.2f} and {upper.value:.2f}.")
    return (bounds,)


@app.cell
def _(age, bounds, inputs, mo, one, pool, spin):
    mo.stop(not one.value, mo.md("Press *Spin 1 life* to play out one life."))
    life = spin(inputs, age.value, pool.value, bounds, n_lives=1, seed=None).iloc[0]
    met = round(inputs["curve"][age.value])
    real = f"met a possible partner at age {life.age_met}" if life.found_partner else "did not meet anyone"
    soul = "found your soul mate" if life.found_soul_mate else "did not find your soul mate"
    mo.md(f"""
    By age {age.value} you have met and talked to about {met:,} new people.

    - Real world rules: you {real}.
    - Soul mate rules: you {soul}.
    """)
    return


@app.cell
def _(age, bounds, inputs, many, mo, np, plt, pool, spin):
    mo.stop(not many.value, mo.md("Press *Spin 1,000 lives* to play out a thousand lives."))
    lives = spin(inputs, age.value, pool.value, bounds, n_lives=1000, seed=None)
    partners, soul_mates = int(lives["found_partner"].sum()), int(lives["found_soul_mate"].sum())
    pool_size = inputs["pools"][pool.value][age.value]
    k = np.median(lives["q"] * pool_size / lives["p_recog"])
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(["soul mate rules", "real world rules"], [soul_mates, partners])
    ax.set(xlim=(0, 1000), xlabel="lives out of 1,000 that found someone")
    fig.tight_layout()
    mo.vstack([
        mo.md(
            f"Out of 1,000 lives, {partners:,} met a possible partner under real world rules "
            f"and {soul_mates:,} found their soul mate. The real world rules mean about {k:,.0f} "
            f"possible partners in a pool of {pool_size:,.0f} people."
        ),
        fig,
    ])
    return


if __name__ == "__main__":
    app.run()
