def GaSb(ω):

    import numpy as np

    π = np.pi
    h_bar = 1.054571817E-34
    q = 1.60217663E-19

    eV = ω*h_bar/q

    # model parameters
    E0   = 0.72    #eV
    Δ0   = 1.46-E0 #eV
    E1   = 2.05    #eV
    Δ1   = 2.50-E1 #eV
    E2   = 4.0     #eV
    Eg   = 0.76    #eV
    A    = 0.71    #eV**1.5
    B1   = 6.68
    B11  = 14.29   #eV**-0.5
    Γ    = 0.09    #eV
    C    = 5.69
    γ    = 0.290
    D    = 7.4
    εinf = 1.0

    ω_LO = 0.0292
    ω_TO = 0.0282
    Gamma = 1.1778e-4
    ω_p = 0.0187
    gamma = 1.3018e-3

    def H(x): #Heviside function
        return 0.5 * (np.sign(x) + 1)

    def Epsilon_A(ħω): #E0
        χ0 = ħω/E0
        χso = ħω / (E0+Δ0)
        H0 = H(1-χ0)
        Hso = H(1-χso)
        fχ0 = χ0**-2 * ( 2 -(1+χ0)**0.5 - ((1-χ0)*H0)**0.5 )
        fχso = χso**-2 * ( 2 - (1+χso)**0.5 - ((1-χso)*Hso)**0.5 )
        H0 = H(χ0-1)
        Hso = H(χso-1)
        ε2 = A/(ħω)**2 * ( ((ħω-E0)*H0)**0.5 + 0.5*((ħω-E0-Δ0)*Hso)**0.5)
        ε1 = A*E0**-1.5 * (fχ0+0.5*(E0/(E0+Δ0))**1.5*fχso)
        return ε1 + 1j*ε2

    def Epsilon_B(ħω): #E1
        # ignoring E1+Δ1 contribution - no data on B2 & B21 in the paper
        # result seems to reproduce graphical data from the paper
        χ1 = ħω/E1
        H1 = H(1-χ1)
        ε2 = π*χ1**-2*(B1-B11*((E1-ħω)*H1)**0.5)
        ε2 *= H(ε2) #undocumented trick: ignore negative ε2
        χ1 = (ħω+1j*Γ)/E1
        ε1 = -B1*χ1**-2*np.log(1-χ1**2)
        return ε1.real + 1j*ε2.real

    def Epsilon_C(ħω): #E2
        χ2 = ħω/E2
        ε2 = C*χ2*γ / ((1-χ2**2)**2+(χ2*γ)**2)
        ε1 = C*(1-χ2**2) / ((1-χ2**2)**2+(χ2*γ)**2)
        return ε1 + 1j*ε2

    def Epsilon_D(ħω): #Eg
        # ignoring ħωq - no data in the paper
        # result seems to reproduce graphical data from the paper
        Ech = E1
        χg = Eg/ħω
        χch = ħω/Ech
        Hg = H(1-χg)
        Hch = H(1-χch)
        ε2 = D/ħω**2 * (ħω-Eg)**2 * Hg * Hch
        return 1j*ε2

    εA  = Epsilon_A(eV)
    εB  = Epsilon_B(eV)
    εC  = Epsilon_C(eV)
    εD  = Epsilon_D(eV)
    ε = εA + εB + εC + εD + εinf

    return ε
