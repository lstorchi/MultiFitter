import os
import re
import numpy as np
import commonutil as cu

def progressbar(current, total, bar_length=40):
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * "=" + ">"
    padding = int(bar_length - len(arrow)) * " "
    print(f"\rProgress: [{arrow}{padding}] {int(fraction*100)}%", end="")


if __name__ == "__main__":
    rootpath_raw = "./data/raw"

    # list a dir content
    listing = None
    try:
        listing = os.listdir(rootpath_raw)
    except Exception as e:
        print(f"Error listing directory: {e}")

    print(f"Starting to read raw data from {rootpath_raw}...")
    allvalues_raw = []
    v1v2pair_raw = set()
    for fname in listing:
        # check if fname it  a dir
        if os.path.isdir(rootpath_raw + "/" + fname):
            #print(f"{fname} is a directory")
            #use a regular exprtession to split v11v11 into v11 and v11
            match = re.match(r"v(\d+)v(\d+)", fname)
            if match:            
                v1dir = match.group(1)
                v2dir = match.group(2)
                v1v2pair_raw.add((int(v1dir), int(v2dir)))
                #print(f"v1: {v1dir}, v2: {v2dir}")
                for datafile in os.listdir(rootpath_raw + "/" + fname):
                    #print(f"  {datafile}")
                    viroval = datafile.split(".")[0]
                    #print(f"    viroval: {viroval}")
                    # re to split v11v11j01j01 into v11, v11, j01, j01
                    match2 = re.match(r"v(\d+)v(\d+)j(\d+)j(\d+)", viroval)
                    if match2:
                        v1 = match2.group(1)
                        v2 = match2.group(2)
                        j1 = match2.group(3)
                        j2 = match2.group(4)
                        if v1 != v1dir or v2 != v2dir:
                            print(f"ERROR: {datafile} does not match the pattern v{v1dir}v{v2dir}j{j1}j{j2}")
                            exit(1)
                        fp = os.path.join(rootpath_raw, fname, datafile)
                        infileval = []
                        for line in open(fp):
                            line = line.strip()
                            sline = line.split()
                            if len(sline) == 2:
                                e = float(sline[0])
                                c = float(sline[1])
                                infileval.append((e, c))
                            else:
                                print(f"Warning: {datafile} has a line that does not have 2 values: {line}")
                                continue
                        allvalues_raw.append((int(v1), 
                                        int(v2), 
                                        int(j1), 
                                        int(j2), 
                                        infileval))
                        #print(f"    v1: {v1}, v2: {v2}, j1: {j1}, j2: {j2}")
            else:
                print(f"Warning: {fname} does not match the pattern v{v1dir}v{v2dir}")
        else:
            print(f"Warning: {fname} is a file") 
    print(f"Finished reading raw data from {rootpath_raw}. Total v1v2 pairs: {len(v1v2pair_raw)}")

    rootpath_fitted = "./data/fitted"
    LINEDIM = 20
    # list a dir content
    listing = None
    try:
        listing = os.listdir(rootpath_fitted)
    except Exception as e:
        print(f"Error listing directory: {e}")

    print(f"Starting to read fitted data from {rootpath_fitted}...")
    allvalues_fitted = []
    v1v2pair_fitted = set()
    for fname in listing:
        # check if fname it  a dir
        if os.path.isdir(rootpath_fitted + "/" + fname):
            #print(f"{fname} is a directory")
            #use a regular exprtession to split v11v11 into v11 and v11
            match = re.match(r"v(\d+)v(\d+)", fname)
            if match:            
                v1dir = match.group(1)
                v2dir = match.group(2)
                v1v2pair_fitted.add((int(v1dir), int(v2dir)))
                #print(f"v1: {v1dir}, v2: {v2dir}")
                for datafile in os.listdir(rootpath_fitted + "/" + fname):
                    #print(f"  {datafile}")
                    viroval = datafile.split(".")[0]
                    #print(f"    viroval: {viroval}")
                    # re to split v11v11j01j01 into v11, v11, j01, j01
                    match2 = re.match(r"v(\d+)v(\d+)j(\d+)j(\d+)", viroval)
                    if match2:
                        v1 = match2.group(1)
                        v2 = match2.group(2)
                        j1 = match2.group(3)
                        j2 = match2.group(4)
                        if v1 != v1dir or v2 != v2dir:
                            print(f"ERROR: {datafile} does not match the pattern v{v1dir}v{v2dir}j{j1}j{j2}")
                            exit(1)
                        fp = os.path.join(rootpath_fitted, fname, datafile)
                        infileval = []
                        line = open(fp).readlines()
                        if len(line) != 1:
                            print(f"Warning: {datafile} has more than 1 line, only the first line will be read")
                        line = line[0].strip()
                        slines = line.split()
                        if len(slines) != LINEDIM:
                            print(f"Warning: {datafile} does not have {2 * LINEDIM} values, it has {len(slines)} values"    )
                        else:
                            a1 = int(slines[0])
                            assert a1 == int(v1)
                            a2 = int(slines[1])
                            assert a2 == int(v2)
                            a3 = int(slines[2])
                            assert a3 == int(j1)
                            a4 = int(slines[3])
                            assert a4 == int(j2)
                            coeffs = []
                            for v in slines[4:]:
                                coeffs.append(float(v.replace('D', 'E').replace('d', 'e')))
                        
                        allvalues_fitted.append((int(v1), 
                                        int(v2), 
                                        int(j1), 
                                        int(j2), 
                                        coeffs))
                        #print(f"    v1: {v1}, v2: {v2
                            
            else:
                print(f"Warning: {fname} does not match the pattern v{v1dir}v{v2dir}")
        else:
            print(f"Warning: {fname} is a file") 
    print(f"Finished reading fitted data from {rootpath_fitted}. Total v1v2 pairs: {len(v1v2pair_fitted)}")

    assert v1v2pair_raw == v1v2pair_fitted, f"v1v2pair_raw: {v1v2pair_raw}, v1v2pair_fitted: {v1v2pair_fitted}"

    v1v2pair = sorted(list(v1v2pair_raw))
    for v1, v2 in v1v2pair:
        print(f"v1: {v1}, v2: {v2}")

    print(f"Total v1v2 pairs: {len(v1v2pair)}")
    print("Starting to organize data for modelling...")
    data = {}
    for v1, v2 in v1v2pair:
        raw_j1j2 = []
        for iv1, iv2, ij1, ij2, infileval in allvalues_raw:
            if iv1 == v1 and iv2 == v2:
                raw_j1j2.append((ij1, ij2))
        
        fitted_j1j2 = []
        for iv1, iv2, ij1, ij2, coeffs in allvalues_fitted:
            if iv1 == v1 and iv2 == v2:
                fitted_j1j2.append((ij1, ij2))

        assert set(raw_j1j2) == set(fitted_j1j2), f"v1: {v1}, v2: {v2}, raw_j1j2: {raw_j1j2}, fitted_j1j2: {fitted_j1j2}" 

        for j1, j2 in raw_j1j2:
            data[(v1, v2, j1, j2)] = {}

    for v1, v2, j1, j2  in data.keys():
        for iv1, iv2, ij1, ij2, infileval in allvalues_raw:
            if iv1 == v1 and iv2 == v2 and ij1 == j1 and ij2 == j2:
                data[(v1, v2, j1, j2)]["raw"] = infileval
                break

        for iv1, iv2, ij1, ij2, coeffs in allvalues_fitted:
            if iv1 == v1 and iv2 == v2 and ij1 == j1 and ij2 == j2:
                data[(v1, v2, j1, j2)]["fitted"] = coeffs
                break
    print(f"Finished organizing data for modelling. Total data points: {len(data)}")

    print("Starting to generate fitted curves and organize data for modelling...")
    Xraw = []
    Xfit = []
    yraw = []
    yfit = []
    for v1, v2,j1, j2 in data.keys():
        rawe = [x[0] for x in data[(v1, v2, j1, j2)]["raw"]]
        rawc = [x[1] for x in data[(v1, v2, j1, j2)]["raw"]]
        icoeffs = data[(v1, v2, j1, j2)]["fitted"]
        e0 = icoeffs[0] 
        coeffs = icoeffs[1:]
        fite, fitc, cd = cu.generate_fitted_curve(e0, coeffs) 

        Xfit.append([v1, v2, j1, j2, fite])
        yfit.append(fitc)
        Xraw.append([v1, v2, j1, j2, rawe])
        yraw.append(rawc)
        progressbar(len(Xraw), len(data))
    print(f"\nFinished generating fitted curves and organizing data for modelling. Total data points: {len(Xraw)}")

    print("Starting to flatten data for modelling...")
    X_flat = []
    y_flat = []
    # Iterate through your nested lists
    for x_row, y_list in zip(Xraw, yraw):
        v1, v2, j1, j2, energies = x_row
        
        # Pair each individual energy point with its specific cross-section
        for e, c in zip(energies, y_list):
            X_flat.append([v1, v2, j1, j2, e])
            y_flat.append(c)
        progressbar(len(X_flat), len(Xraw) * len(energies)) 
    print()

    Xraw = np.array(X_flat)
    yraw = np.array(y_flat)

    X_flat = []
    y_flat = []

    # Iterate through your nested lists
    for x_row, y_list in zip(Xfit, yfit):
        v1, v2, j1, j2, energies = x_row
        
        # Pair each individual energy point with its specific cross-section
        for e, c in zip(energies, y_list):
            X_flat.append([v1, v2, j1, j2, e])
            y_flat.append(c)
        progressbar(len(X_flat), len(Xfit) * len(energies))
    print()

    Xfit = np.array(X_flat)
    yfit = np.array(y_flat)
    print(f"Finished flattening data for modelling. Total data points: {len(Xraw)}")

    print("Saving data for modelling...")
    np.savez("modelling_data.npz", 
             Xraw=Xraw, yraw=yraw, 
             Xfit=Xfit, yfit=yfit)

