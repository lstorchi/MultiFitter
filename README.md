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

GLOBAL MODEL

buildglobalmodel.py : run and build a global model using alla data 
testglobalmodel.ipynb : read the results of testmodelling.py and run the prediction 
