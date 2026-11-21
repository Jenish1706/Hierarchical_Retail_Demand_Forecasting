# fig11_horizon_degradation

WAPE at each day of the forecast horizon, top 8 models by mean accuracy. Flattest: lightgbm_tweedie (-0.110 WAPE/day). Steepest: seasonal_naive_weekly (+0.159 WAPE/day). Methods that emit a constant demand rate (Croston, ADIDA, SES) are flat by construction, since their forecast does not depend on how far ahead it is made. A rising curve indicates a model exploiting short-horizon information that decays -- useful for next-week replenishment, less so at four weeks.
