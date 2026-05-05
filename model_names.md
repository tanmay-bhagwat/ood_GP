Train and test distribution graphs in ppt_imgs.
Remember to check that min_var is set to 0.02 (or some low fixed const) so that you can compare pred-vs-true graphs with different descriptors.

## checkpoint_m1 (z=0.1, MAE=0.0476)
Remove noise from GPvar-error plot to see the cloud. With noise, its just a thin line.
Use this to explain high density of train points and thus, interpolation to test points.

### ln=-6.5, ls=3.5, ll=0.05
Add in test structures from norm_y>=2.5 to see the tilt of higher error values towards diagonal.
I think 50-50 to 80-20 would work well as graph to show the tilt of higher energies towards diag.

### ln=-8, ls=-2, ll=0.05
For 5% from train dist (95-5) you can see the large spread+var in high energy (above 2.5).


## checkpoint_m2
Reduced sigma=0.5 for the SOAP descriptor to see if I can get the model to differentiate between test points even in the interpolation regime. With frozen weights, ln=-10, ll=-1, ls=-3 I can clearly see the high error points tilt towards the diag but the low error ones do not.

## checkpoint m3
Descriptors and model from sigma=0.5, r_cut=6 SOAP descriptors after FPS. Priors added for all hyperparams.

## checkpoint m4
FPS descriptors from m3 and 32 lengthscales rather than just one. Also implemented two-stage training where embedding params are trained only after 150 epochs.