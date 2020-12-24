# Critically analyzing the performance of (supposedly) state-of-the-art blind source separation systems.

SOTA source separation systems are typically evaluated on a single metric: SDR. While SDR is important, and a high SDR can signify a well-separated source, the methods of aggregation of SDR across source separation systems can vary immensely, as well as SDR not being a metric of source quality, or human perception. Windowing in SDR can also lead to unsatsifactory evaluation on specific sources,as well as ignoring outliers.

This can lead researchers to persue SOTA on the metric of SDR, without creating a source separation systems that actually improves separation as perceived by humans. We look at [papers with code](https://paperswithcode.com/sota/music-source-separation-on-musdb18])(PWC) for models that persue this SOTA status.

## Pretrained models.
We evaluate the pretrained models, Demucs, Conv-TasNet, Wave-U-Net, and OpenUnmix. For Demucs and Conv-TasNet, we use the "extra" versions from the repository: https://github.com/facebookresearch/demucs

## Our evaluation method
Since evaluation and aggregation methods may differ from paper to paper, we define our evaluation and aggregation here:
1. Create chunks of each stem in the MUSDB18 test set. Each chunk is 8 seconds long, with a 4 second hop between each chunk.
2. Remove all chunks where at least one source is silent. This is done by finding the reference chunk with the peak power for a source, then labeling all chunks that have power less than 8db than the reference to be silent.
3. Input the mix of each chunk into the network, and measure the metrics of the separated sources, including but not limited to, SI-SDR, SDR, SI-SDRi, SAR, SI-SAR. We only calculate these metrics on a single window for our final aggregations.

## Our aggregation method
We wish to get the following values for each model: Mean and Median metric for each source, and Mean and Median metric over all sources. We call these the source-specific metrics, and the source-aggregated metrics, respectively.
### Source-specific metrics
To calculate the mean/median source-specific metric, find the mean/median of the recorded metric of each chunk.

For example, to get the mean SDR of the `drums` source, we find the SDR for each `drums` source for each chunk, then we find the means of those SDRs.
### Source-aggregated metric
To calculate the mean/median source-aggregated metric, we first find the mean metric over all source for each chunk, then find the mean/median of those means.

For example, if we wanted to find the median SI-SDR over all sources over the MUSDB18 test set, we would take the following steps:
1. Find the mean SI-SDR of each chunk. This involves finding the SI-SDR for each of the four sources, `drums`, `other`, `vocals` and `bass`, then calculating the mean of them. This is the SI-SDR of all sources in this chunk.
2. Find the median SI-SDR of each mean.

Why is a mean used to find metrics over all sources? This is because median will ignore that best and worst source, which is problematic for a dataset of 4 items.

# Findings in Jupyter Notebooks so far
1. In the context of ranking models on mean metric, SDR, SI-SDR, and SI-SDRi are result in the same ranking. This is also true when median is used.
2. When examining vocals SDR, Conv-TasNet beats Demucs by about .22 SDR, which is not the case in papers with code, while in PWC, Demucs beats Conv-TasNet by .31
3. In estimated separated sources, a fail case is the model generating approximately silent sources. This results in approximately 0 SDR, but a large negative SI-SDR. In aggregations, this results in SI-SDR appearing lower than SDR. This may incentivize systems that result in silence for mixes they know to be difficult to separate.
4. Windowing, especially when a source's attacks are sparse in a signal, may result in far lower SDRs as a result of a ground truth window that is completely silent while the estimated window has neglible sound.
5. There do not appear to be any specific outlier songs that affect the rankings of the models we are studying. While there do exists songs that are difficult to separate, they are difficult for most models. 
