def sel_em_TPV(N_a,N_d,wp_eS,gam_eS,wp_cS,gam_cS):

    N_a = N_a.cpu().item()
    N_d = N_d.cpu().item()
    wp_eS = wp_eS.cpu().item()
    gam_eS = gam_eS.cpu().item()
    wp_cS = wp_cS.cpu().item()
    gam_cS = gam_cS.cpu().item()

    from MESH import SimulationPlanar
    import numpy as np

    #Constants
    h_bar = 1.054571817e-34
    k_b = 1.380649e-23
    c = 2.99792458e+8
    q = 1.60217663e-19
    e_v = 8.85e-14

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
    eps_cS = np.zeros(shape=(1999,3))

    #Material properties
    e_s = 16.2*e_v
    N_i = 2e13

    u_e = 1e2
    u_h = 1e2

    V_bi = k_b*T_c/q * np.log(N_a*N_d/N_i**2)
    t_c = 0.01*(2*e_s/q*V_bi*(1/N_a+1/N_d))**0.5

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

        eps_cS[i,0] = w
        eps_cS[i,1] = np.real(Drude(w,wp_cS,gam_cS))
        eps_cS[i,2] = np.imag(Drude(w,wp_cS,gam_cS))

    np.savetxt("eSubstrate.txt",eps_eS)
    np.savetxt("Vacuum.txt",eps_vac)
    np.savetxt("cSubstrate.txt",eps_cS)

    # Setting up MESH ---------------------------------------------------------------------------

    sim = SimulationPlanar()
    sim.AddMaterial('eSubstrate','eSubstrate.txt')
    sim.AddMaterial('Vacuum','Vacuum.txt')
    sim.AddMaterial('Ge','Ge.txt')
    sim.AddMaterial('cSubstrate','cSubstrate.txt')

    sim.AddLayer(layer_name = 'eSubstrate', thickness = 0, material_name = 'eSubstrate')
    sim.AddLayer(layer_name = 'Vacuum', thickness = d_v, material_name = 'Vacuum')
    sim.AddLayer(layer_name = 'Ge', thickness = t_c, material_name = 'Ge')
    sim.AddLayer(layer_name = 'cSubstrate', thickness = 0, material_name = 'cSubstrate')

    sim.OptUseQuadgk()
    sim.SetKParallelIntegral(integral_end = 600)
    sim.SetThread(num_thread = 48)

    # Analysis ----------------------------------------------------------------------------------

    sim.SetSourceLayer(layer_name = 'eSubstrate')
    sim.SetProbeLayer(layer_name = 'Ge')

    sim.SetProbeLayerZCoordinate(0)
    sim.InitSimulation()
    sim.IntegrateKParallel()
    phi1 = sim.GetPhi()

    sim.SetProbeLayerZCoordinate(t_c)
    sim.InitSimulation()
    sim.IntegrateKParallel()
    phi2 = sim.GetPhi()

    for i in range(1999):
        q_w_1[i] = (phi1[i] - phi2[i]) * theta[i]
        q_w_2[i] = phi1[i] * theta[i]

    np.savetxt("q_w_1.txt",q_w_1)
    np.savetxt("q_w_2.txt",q_w_2)

    # Postprocessing ---------------------------------------------------------------------------------

    integral = []

    for j in range(333):
        integral.append(3*dw/8/h_bar*(q_w_1[3*j+999]/W[3*j+999] + 3*q_w_1[3*j+1000]/W[3*j+1000] + 3*q_w_1[3*j+1001]/W[3*j+1001] + q_w_1[3*j+1002]/W[3*j+1002]))

    J_ph = q*sum(integral)

    integral = []

    for j in range(666):
        integral.append(3*dw/8*(q_w_2[3*j]+3*q_w_2[3*j+1]+3*q_w_2[3*j+2]+q_w_2[3*j+3]))

    P_r = sum(integral)

    #Electrical analysis

    J_s = (q*N_i**2*u_e/N_a + q*N_i**2*u_h/N_d)*1e4
    V_oc = k_b*T_c/q*np.log(J_ph/J_s+1)

    dV = V_oc/1000
    P_e = 0

    for i in range(1000):
        V = i*dV
        P2 = V*(J_ph-J_s*(np.exp(q*V/k_b/T_c)-1))
        if P2 > P_e:
            P_e = P2

    eff = P_e/P_r
    print(eff)
    #print(P_e)
    return eff