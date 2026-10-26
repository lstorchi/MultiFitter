initialtest.ipynb: parses raw cross-section data and Fortran-formatted 
                   Legendre polynomial coefficients, mathematically
                   reconstructs the excitation functions, and plots the
                   resulting fitted curves alongside the original raw data
                   points for visual verification.

preparedata.py: processes raw and coefficient-based molecular transition
                data, reconstructs the fitted excitation curves, and
                flattens the nested structures into machine-learning-ready
                feature and target arrays saved as a compressed .npz
                archive.

                this produce  modelling_data.npz

PER v1v2 models

1. Refitted NN model
    
    Feedforward neural network (MLP) regressor that is pre-trained on fitted data
    and subsequently fine-tuned on raw data with its earlier layers frozen.

    NN_build_refitted_perv1v2.py : build model
    NN_test_refitted_perv1v2.py : read and test model

2. Two Stages NN model

    A base network (64-64-1 with softplus output) is trained on the fitted curves to learn the
    smooth functional-fit surface, and a small, zero-initialized correction network (32-32-1 with 
    L2 regularization) is then trained on the raw data to learn the relative deviation 
    r = (raw − fit)/(fit + s), so that the final prediction is ŷ = y_base + (y_base + s)·Δ, 
    clipped at zero.

    NN_build_twostages_perv1v2.py : build model
    NN_test_twostages_perv1v2.ipynb : read and test model

3.  GPR

    We use a two-stage GPR model. First, PCA reduces each fitted curve to a few component 
    coefficients, and one Gaussian process per component predicts its coefficient from (j1, 
    j2), so any curve, including new ones, is rebuilt as the mean curve plus the predicted 
    components. Second, a Gaussian process over (j1, j2, E) learns the relative deviation of 
    the raw data from the fit, r = (raw − fit)/(fit + s), and the final prediction is ŷ = 
    y_base + (y_base + s)·Δ, clipped at zero, with uncertainty bands available from both stages.

    GPR_build_perv1v2.py : build model
    GPR_test_perv1v2.py : test model