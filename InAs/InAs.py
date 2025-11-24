def InAs(?):

    import numpy as np

    ? = np.pi
    h_bar = 1.054571817E-34
    q = 1.60217663E-19

    eV = ?*h_bar/q

    # model parameters
    E0   = 0.36    #eV
    ?0   = 0.76-E0 #eV
    E1   = 2.50    #eV
    ?1   = 2.78-E1 #eV
    E2   = 4.45    #eV
    Eg   = 1.07    #eV
    A    = 0.61    #eV**1.5
    B1   = 6.59
    B11  = 13.76   #eV**-0.5
    ?    = 0.21    #eV
    C    = 1.78
    ?    = 0.108
    D    = 20.8
    ?inf = 2.8

    def H(x): #Heviside function
        return 0.5 * (np.sign(x) + 1)

    def Epsilon_A(h?): #E0
        ?0 = h?/E0
        ?so = h? / (E0+?0)
        H0 = H(1-?0)
        Hso = H(1-?so)
        f?0 = ?0**-2 * ( 2 -(1+?0)**0.5 - ((1-?0)*H0)**0.5 )
        f?so = ?so**-2 * ( 2 - (1+?so)**0.5 - ((1-?so)*Hso)**0.5 )
        H0 = H(?0-1)
        Hso = H(?so-1)
        ?2 = A/(h?)**2 * ( ((h?-E0)*H0)**0.5 + 0.5*((h?-E0-?0)*Hso)**0.5)
        ?1 = A*E0**-1.5 * (f?0+0.5*(E0/(E0+?0))**1.5*f?so)
        return ?1 + 1j*?2

    def Epsilon_B(h?): #E1
        # ignoring E1+?1 contribution - no data on B2 & B21 in the paper
        # result seems to reproduce graphical data from the paper
        ?1 = h?/E1
        H1 = H(1-?1)
        ?2 = ?*?1**-2*(B1-B11*((E1-h?)*H1)**0.5)
        ?2 *= H(?2) #undocumented trick: ignore negative ?2
        ?1 = (h?+1j*?)/E1
        ?1 = -B1*?1**-2*np.log(1-?1**2)
        return ?1.real + 1j*?2.real

    def Epsilon_C(h?): #E2
        ?2 = h?/E2
        ?2 = C*?2*? / ((1-?2**2)**2+(?2*?)**2)
        ?1 = C*(1-?2**2) / ((1-?2**2)**2+(?2*?)**2)
        return ?1 + 1j*?2

    def Epsilon_D(h?): #Eg
        # ignoring h?q - no data in the paper
        # result seems to reproduce graphical data from the paper
        Ech = E1
        ?g = Eg/h?
        ?ch = h?/Ech
        Hg = H(1-?g)
        Hch = H(1-?ch)
        ?2 = D/h?**2 * (h?-Eg)**2 * Hg * Hch
        return 1j*?2

    ?A  = Epsilon_A(eV)
    ?B  = Epsilon_B(eV)
    ?C  = Epsilon_C(eV)
    ?D  = Epsilon_D(eV)
    ? = ?A + ?B + ?C + ?D + ?inf

    return ?