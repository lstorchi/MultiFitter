def legpol(xx, nleg):
    """
    Calculates Legendre polynomials at point xx.
    nleg = number of coefficients (exponents from 0 to nleg-1).
    """
    pol = [0.0] * nleg
    
    if nleg > 0:
        pol[0] = 1.0  # Equivalent to p(0)=1.d0 [cite: 4]
    if nleg > 1:
        pol[1] = xx   # Equivalent to p(1)=xx [cite: 4]
        
    for i in range(2, nleg):
        # Recurrence relation for Legendre polynomials [cite: 5]
        pol[i] = ((2 * i - 1) * xx * pol[i-1] - (i - 1) * pol[i-2]) / i
        
    return pol


def legder(xx, nleg):
    """
    Calculates the derivatives of Legendre polynomials at point xx.
    nleg = number of coefficients (exponents from 0 to nleg-1).
    """
    p = [0.0] * nleg
    der = [0.0] * nleg
    
    if nleg > 0:
        p[0] = 1.0    # Equivalent to p(0)=1.d0 [cite: 5]
        der[0] = 0.0  # Equivalent to d(0)=0.d0 [cite: 5]
    if nleg > 1:
        p[1] = xx     # Equivalent to p(1)=xx [cite: 5]
        der[1] = 1.0  # Equivalent to d(1)=1.d0 [cite: 6]
        
    for i in range(2, nleg):
        # Polynomial recurrence relation [cite: 6]
        p[i] = ((2 * i - 1) * xx * p[i-1] - (i - 1) * p[i-2]) / i
        
        # Derivative recurrence relation [cite: 6]
        der[i] = ((2 * i - 1) * (p[i-1] + xx * der[i-1]) - (i - 1) * der[i-2]) / i
        
    return der


def generate_fitted_curve(e0, coeffs, e1=0.0, e2=900.0):
    """
    Generates the excitation function and derivative over an energy grid.
    Translates the Fortran loops and uses the legpol/legder functions.
    """
    nleg = len(coeffs)
    delta = e2 - e1
    
    # Calculate starting and ending indices based on e0 
    ie0 = int(round(e0 * 1000.0))
    ie1 = int(round(e0 * 10.0)) + 1
    ie2 = ie1 * 100 - 1
    
    energies = []
    csfits = []
    csders = []
    
    # --- FIRST LOOP: Fine energy grid (step of 0.001) --- 
    # Fortran: do ie=ie0,ie2 
    for ie in range(ie0, ie2 + 1):
        e = ie * 0.001  # 
        
        # Calculate normalized variable xx 
        xx = (2.0 * e - e2 - e1) / delta  # 
        
        # Get polynomials and derivatives
        pol = legpol(xx, nleg)
        der = legder(xx, nleg)
        
        # Calculate fit and derivative 
        csfit = sum(c * p for c, p in zip(coeffs, pol))
        csder = sum(c * d for c, d in zip(coeffs, der))
        
        # Scale the derivative [cite: 3]
        csder = csder * 2.0 / delta  # [cite: 3]
        
        energies.append(e)
        csfits.append(csfit)
        csders.append(csder)

    # --- SECOND LOOP: Coarse energy grid (step of 0.1) --- [cite: 3]
    # Fortran: do ie=ie1,9000 [cite: 3]
    for ie in range(ie1, 9000 + 1):
        e = ie * 0.1  # [cite: 3]
        
        # Calculate normalized variable xx [cite: 3]
        xx = (2.0 * e - e2 - e1) / delta  # [cite: 3]
        
        # Get polynomials and derivatives
        pol = legpol(xx, nleg)
        der = legder(xx, nleg)
        
        # Calculate fit and derivative [cite: 3, 4]
        csfit = sum(c * p for c, p in zip(coeffs, pol))
        csder = sum(c * d for c, d in zip(coeffs, der))
        
        # Scale the derivative [cite: 4]
        csder = csder * 2.0 / delta  # [cite: 4]
        
        energies.append(e)
        csfits.append(csfit)
        csders.append(csder)

    return energies, csfits, csders
