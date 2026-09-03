# Modified_pygwb
This repo contains the code of the modified version of pygwb package used to produce the results shown in arxiv 2606.23816.

One can generate noise-only and SGWB-only GW frames by running `Simulation_scripts/generate_pure_noise_frames.py` and `generate_SGWB_frames.py`, respectively. 

The `Plotting_scripts.ipynb` is the jupyter notebook to reproduce the plots we show in the paper. The output of Fig. 15 will be different when different random seeds are used, while the purpose of this figure is to show that the new expression of the bias factor is more consistent with the data. One needs `bias_data_paper.npz` for the last figure, which is ~1 GB, so we do not incorporate that in the `/Data` folder. 

`Modified_pygwb` folders contains modified files of the `pygwb` package. 