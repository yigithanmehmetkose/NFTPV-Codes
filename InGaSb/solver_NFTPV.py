def sel_em_TPV(x,N_a,N_d,wp_eS,gam_eS,wp_cS,gam_cS):

    #x = x.cpu().item()
    #N_a = N_a.cpu().item()
    #N_d = N_d.cpu().item()
    #wp_eS = wp_eS.cpu().item()
    #gam_eS = gam_eS.cpu().item()
    #wp_cS = wp_cS.cpu().item()
    #gam_cS = gam_cS.cpu().item()

    from MESH import SimulationPlanar
    from InSb import InSb
    from GaSb import GaSb
    import numpy as np

    #Constants
    h_bar = 1.054571817E-34
    k_b = 1.380649E-23
    c = 2.99792458E8
    q = 1.60217663E-19
    e_v = 8.85E-14
    m0 = 9.109E-31

    #Inputs
    T_e = 1000
    d_v = 100e-9
    T_c = 300

    #Preprocessing
    N_a = 10**N_a
    N_d = 10**N_d

    dw = 1E12
    W = [dw*(i+1) for i in range(1999)]

    theta = np.zeros(1999)
    q_w_1 = np.zeros(1999)
    q_w_2 = np.zeros(1999)

    eps_eS = np.zeros(shape=(1999,3))
    eps_vac = np.zeros(shape=(1999,3))
    eps_InGaSb = np.zeros(shape=(1999,3))
    eps_cS = np.zeros(shape=(1999,3))

    #Material properties
    E_g = (x*0.726 + (1-x)*0.17 - x*(1-x)*0.4133)*q
    print(E_g/q)
    e_s = (16.8-1.1*x)*e_v

    mstar_e = (0.015 + 0.01*x + 0.025*x**2)*m0
    mstar_h = (0.43 - 0.03*x)*m0
    N_c = 2*(mstar_e*k_b*T_c/(2*np.pi*h_bar**2))**1.5
    N_v = 2*(mstar_h*k_b*T_c/(2*np.pi*h_bar**2))**1.5
    N_i = (N_c*N_v)**0.5*np.exp(-E_g/(2*k_b*T_c))/1e6

    u_e = 1e2
    u_h = 1e2

    if N_a*N_d > N_i**2:
        V_bi = k_b*T_c/q * np.log(N_a*N_d/N_i**2)
        t_c = 0.01*(2*e_s/q*V_bi*(1/N_a+1/N_d))**0.5
    else:
        t_c = 5e-9

    #Drude Model
    def Drude(w,wp,gam):
        eps = 1-wp**2/(w*(w+1j*gam))
        return eps

    # Material Properties -----------------------------------------------------------------------

    for i in range(1999):
        w = W[i]
        theta[i] = h_bar*w/(np.exp(h_bar*w/k_b/T_e)-1)

        eps_eS[i,0] = w
        eps_eS[i,1] = np.real(Drude(w,wp_eS,gam_eS))
        eps_eS[i,2] = np.imag(Drude(w,wp_eS,gam_eS))

        eps_vac[i,0] = w
        eps_vac[i,1] = 1
        eps_vac[i,2] = 1e-18

        eps_InGaSb[i,0] = w
        eps_InGaSb[i,1] = x*np.real(GaSb(w)) + (1-x)*np.real(InSb(w))
        eps_InGaSb[i,2] = x*np.imag(GaSb(w)) + (1-x)*np.imag(InSb(w))

        eps_cS[i,0] = w
        eps_cS[i,1] = np.real(Drude(w,wp_cS,gam_cS))
        eps_cS[i,2] = np.imag(Drude(w,wp_cS,gam_cS))

    np.savetxt("eSubstrate.txt",eps_eS)
    np.savetxt("Vacuum.txt",eps_vac)
    np.savetxt("InGaSb.txt",eps_InGaSb)
    np.savetxt("cSubstrate.txt",eps_cS)

    # Setting up MESH ---------------------------------------------------------------------------

    sim = SimulationPlanar()
    sim.AddMaterial('eSubstrate','eSubstrate.txt')
    sim.AddMaterial('Vacuum','Vacuum.txt')
    sim.AddMaterial('InGaSb','InGaSb.txt')
    sim.AddMaterial('cSubstrate','cSubstrate.txt')

    sim.AddLayer(layer_name = 'eSubstrate', thickness = 0, material_name = 'eSubstrate')
    sim.AddLayer(layer_name = 'Vacuum', thickness = d_v, material_name = 'Vacuum')
    sim.AddLayer(layer_name = 'InGaSb', thickness = t_c, material_name = 'InGaSb')
    sim.AddLayer(layer_name = 'cSubstrate', thickness = 0, material_name = 'cSubstrate')

    sim.OptUseQuadgk()
    sim.SetKParallelIntegral(integral_end = 600)
    sim.SetThread(num_thread = 48)

    # Analysis ----------------------------------------------------------------------------------

    #Power absorbed by the cell

    sim.SetProbeLayer(layer_name = 'InGaSb')

    sim.SetProbeLayerZCoordinate(0)
    sim.SetSourceLayer(layer_name = 'eSubstrate')
    sim.InitSimulation()
    sim.IntegrateKParallel()
    phi1 = sim.GetPhi()

    sim.SetProbeLayerZCoordinate(t_c)
    sim.SetSourceLayer(layer_name = 'eSubstrate')
    sim.InitSimulation()
    sim.IntegrateKParallel()
    phi2 = sim.GetPhi()

    for i in range(1999):
        q_w_1[i] = (phi1[i] - phi2[i]) * theta[i]
        q_w_2[i] = phi1[i] * theta[i]

    np.savetxt("q_w_1.txt",q_w_1)
    np.savetxt("q_w_2.txt",q_w_2)

    # Postprocessing ---------------------------------------------------------------------------------

    i_g = int(np.floor(E_g/h_bar/dw)) - 1
    N_int = int(np.floor((1995-i_g)/3))
    integral = []

    for j in range(N_int):
        integral.append(3*dw/8/h_bar*(q_w_1[3*j+i_g]/W[3*j+i_g] + 3*q_w_1[3*j+i_g+1]/W[3*j+i_g+1] + 3*q_w_1[3*j+i_g+2]/W[3*j+i_g+2] + q_w_1[3*j+i_g+3]/W[3*j+i_g+3]))

    J_ph = q*sum(integral)

    integral = []

    for j in range(666):
        integral.append(3*dw/8*(q_w_2[3*j]+3*q_w_2[3*j+1]+3*q_w_2[3*j+2]+q_w_2[3*j+3]))

    P_r = sum(integral)

    #Electrical analysis

    J_s = (q*N_i**2*u_e/N_a + q*N_i**2*u_h/N_d)*1e4

    if J_s > J_ph:
        print("ABNORMAL CONDITION")
        J_s = J_ph*0.1

    V_oc = k_b*T_c/q*np.log(J_ph/J_s+1)

    dV = V_oc/1000
    P_e = 0

    for i in range(1000):
        V = i*dV
        P2 = V*(J_ph-J_s*(np.exp(q*V/k_b/T_c)-1))
        if P2 > P_e:
            P_e = P2

    eff = P_e/P_r

    if np.isnan(eff):
        eff = 0.0

    print(eff)
    print(P_e)

    return eff