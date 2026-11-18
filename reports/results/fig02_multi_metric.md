# fig02_multi_metric

Each metric normalised to its own best model, so 1.0 is the best score on that metric. Where a model's bars differ in height across metrics, the metrics disagree about it -- MASE is less sensitive to a few large misses than RMSSE, so a model better on MASE and worse on RMSSE is winning on typical days and losing on spikes.
