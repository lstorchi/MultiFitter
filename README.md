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

1. Refitted model
    
    Feedforward neural network (MLP) regressor that is pre-trained on fitted data
    and subsequently fine-tuned on raw data with its earlier layers frozen.

    NN_build_refitted_perv1v2.py : build model
    NN_test_refitted_perv1v2.py : read and test model
