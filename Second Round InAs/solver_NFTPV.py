def sel_em_TPV(N_a,N_d):

    from MESH import SimulationPlanar
    from InAs import InAs
    import numpy as np
    from Tungsten import Tungsten as Tungsten2

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
    eps_InAs = np.zeros(shape=(1999,3))
    eps_cS = np.zeros(shape=(1999,3))

    #Material properties
    e_s = 15.15*e_v
    N_i = 1e15

    u_e = 1e2
    u_h = 1e2

    V_bi = k_b*T_c/q * np.log(N_a*N_d/N_i**2)
    t_c = 0.01*(2*e_s/q*V_bi*(1/N_a+1/N_d))**0.5

    #Copper
    def Cu(w):
        eps = 1-11.23e15**2/(w*(w+1j*138e11))
        return eps

    #Tungsten
    def Tungsten(w_rad_s,T_K):

        h_bar = 1.054571817e-34
        q = 1.60217663e-19

        w_eV = h_bar * w_rad_s / q

        T_room = 300
        alpha = 3.5e-2
        beta = 2.5e-3

        f0 = 0.275
        gamma0 = 0.05 + alpha * (np.sqrt(T_K) - np.sqrt(T_room))
        w_p = 1.75 / np.sqrt(1 + 4.3e-6 * (T_K - T_room))

        f_i = np.array([0.06, 0.19, 0.75, 2.39])
        w_i0 = np.array([0.94, 1.86, 3.35, 7.7])
        gamma_i0 = np.array([0.72, 1.33, 2.44, 2.8])

        gamma_i = gamma_i0 + alpha * (np.sqrt(T_K) - np.sqrt(T_room))
        w_i = np.sqrt(w_i0**2 - beta * (np.sqrt(T_K) - np.sqrt(T_room)))

        eps_Drude = f0 * w_p**2 / (w_eV**2 + 1j * w_eV * gamma0)
        eps_Lorentz = np.zeros_like(w_eV, dtype=np.complex128)

        for j in range(len(f_i)):
            eps_Lorentz += (f_i[j] * w_p**2 / (w_i[j]**2 - w_eV**2 - 1j * w_eV * gamma_i[j]))

        eps = 1 - eps_Drude + eps_Lorentz
        return eps

    #Indium-tin-oxide
    eps_inf_ITO = 4
    gam_ITO = 2.4050e+14
    w_p_ITO = np.sqrt(eps_inf_ITO*(1.3370e+15**2+gam_ITO**2))

    def ITO(w):
        eps = eps_inf_ITO - w_p_ITO**2/(w**2+1j*w*gam_ITO)
        return eps

    # Material Properties -----------------------------------------------------------------------

    for i in range(1999):
        w = W[i]
        theta[i] = h_bar*w/(np.exp(h_bar*w/k_b/T_e)-1)

        eps_eS[i,0] = w
        eps_eS[i,1] = np.real(Tungsten(w,T_e))
        eps_eS[i,2] = np.imag(Tungsten(w,T_e))

        #eps_eS[i,1] = np.real(Tungsten2(w))
        #eps_eS[i,2] = np.imag(Tungsten2(w))

        eps_vac[i,0] = w
        eps_vac[i,1] = 1
        eps_vac[i,2] = 1e-18

        eps_InAs[i,0] = w
        eps_InAs[i,1] = np.real(InAs(w))
        eps_InAs[i,2] = np.imag(InAs(w))

        eps_cS[i,0] = w
        eps_cS[i,1] = np.real(Cu(w))
        eps_cS[i,2] = np.imag(Cu(w))

    np.savetxt("eSubstrate.txt",eps_eS)
    np.savetxt("Vacuum.txt",eps_vac)
    np.savetxt("InAs.txt",eps_InAs)
    np.savetxt("cSubstrate.txt",eps_cS)

    # Setting up MESH ---------------------------------------------------------------------------

    sim = SimulationPlanar()
    sim.AddMaterial('eSubstrate','eSubstrate.txt')
    sim.AddMaterial('Vacuum','Vacuum.txt')
    sim.AddMaterial('InAs','InAs.txt')
    sim.AddMaterial('cSubstrate','cSubstrate.txt')

    sim.AddLayer(layer_name = 'eSubstrate', thickness = 0, material_name = 'eSubstrate')
    sim.AddLayer(layer_name = 'Vacuum', thickness = d_v, material_name = 'Vacuum')
    sim.AddLayer(layer_name = 'InAs', thickness = t_c, material_name = 'InAs')
    sim.AddLayer(layer_name = 'cSubstrate', thickness = 0, material_name = 'cSubstrate')

    sim.OptUseQuadgk()
    sim.SetKParallelIntegral(integral_end = 600)
    sim.SetThread(num_thread = 48)

    # Analysis ----------------------------------------------------------------------------------

    sim.SetProbeLayer(layer_name = 'InAs')

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

    integral = []

    for j in range(486):
        integral.append(3*dw/8/h_bar*(q_w_1[3*j+537]/W[3*j+537] + 3*q_w_1[3*j+538]/W[3*j+538] + 3*q_w_1[3*j+539]/W[3*j+539] + q_w_1[3*j+540]/W[3*j+540]))

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

    J_list = np.zeros(1000)

    for i in range(1000):
        V = i*dV
        J = (J_ph-J_s*(np.exp(q*V/k_b/T_c)-1))
        P2 = V*J
        J_list[i] = J
        if P2 > P_e:
            P_e = P2
            print(V)

    eff = P_e/P_r

    print(eff)
    print(P_e)
    print(V_oc)
    print(J_s)

    np.savetxt("J.txt",J_list)

    return eff, P_e

eff, P_e = sel_em_TPV(16,19)