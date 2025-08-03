import numpy as np
import random
from Channel_Generation_6_S1 import Channel_Generation_6 as CG
class Optimisation:
    def __init__(self,K,M,N,power,X,noise_var):
        self.K = K
        self.M = M
        self.N = N
        self.power = power
        self.X =X
        self.noise_var = noise_var
        
    def No_Optimisation(self):  # H1 for Hd, H2 for Hr and H3 for G
        
        W = np.random.rand(self.M,self.K)  # Initialising the  beamforming vector
        
        # phase0 = np.random.rand(self.N)*2*np.pi # Initialising the phase-shift at RIS
        # v = np.exp(1j*phase0)
        # Theta = np.diag(v) 
        
        #H = H1 + (H2 @ Theta @ H3 )               # Channel Equalisation
        
        return  W  
    
    def AO_Single_User(self,u, H1, H2, H3):  # u is the user for which AO is done
        iter = 0
        # initialising the beamforming vectors:
        w = np.zeros((self.M,self.K),dtype=complex) # as BF is optimised for one user only, size is Mx1
        
        # initialising the phase shifts at RIS
        phase0init = np.eye(self.N)
        v = np.exp(1j*phase0init)
        Theta = np.diag(v.diagonal())
        # print("theta=T",Theta.shape)
        # print("H1=", H1.shape)
        # print("H2=",H2.shape)
        # print("G=",H3.shape)
        
        for iter in range(10):
            #The channel equalisation:
            H = H1 + H2 @ (Theta) @ H3
            #print(H.shape) 
            
            # MRT beamforming
            w_MRT =  (np.conj(H).T)
            w_MRT = w_MRT/ (np.linalg.norm(w_MRT,axis=0))
            w_MRT = w_MRT * np.sqrt(self.power)
            #print(w_MRT.shape)
            
            phase0 = np.angle(H1[u,:] @ w_MRT[:,u]) - np.angle(H2[u,:]) - np.angle(H3[:,:] @ w_MRT[:,u])  # w is changed to w_MRT
            v = np.exp(1j*phase0)
            Theta=(np.diag(v))
            #print(Theta.shape)
            
            # To make the phase0 equal to zero
            # a= -np.angle(H @ w_MRT)
            # w= w_MRT*np.exp(1j*a)
            #print("small w shape=",w.shape)
        
        
        w[:,u] = w_MRT[:,u]

        return w, Theta
    
    def ZF_BF(self, H):
        rates = []
        #noise = 10 ** (-85/10 - 3)
        noise = self.noise_var;
        #print("noise=", noise)
        # zero forcing
        Hscale = H/np.sqrt(noise)
        #print("H =",H)
        #print("Hscale=",Hscale)
        W_ZF = np.linalg.pinv(Hscale)
        W_ZF =  W_ZF/np.linalg.norm(W_ZF,axis =0)
        #print(np.linalg.norm(W_ZF,axis=0))
        W_ZF = np.sqrt(self.power/self.K) * W_ZF
        
        for k in range(self.K):  # Making change here from K to X, if using SUS
            signal_strength = (np.abs(Hscale[k,:] @ W_ZF[:,k]))**2
            #print("Signal strenth=",signal_strength)
            #print("signal=",signal_strength)
            # calculating interference from the other users
            interference = 0
            for j in range(self.K):   # Making change here from K to X, If using SUS
                if j != k:
                    interference += (np.abs(Hscale[k,:] @ W_ZF[:,j]))**2
                    #print("interference zf =",interference)
            #print("Signal strenth=",signal_strength)
            rates.append(np.log2(1+(signal_strength/(1 + interference))))

        ZF_rate = np.sum(rates)   
        return ZF_rate
    
    
    def MRT_BF(self,H, W):
        rates = []
        noise = self.noise_var
        # W = np.conj(H).T
        # W = W/(np.linalg.norm(W, axis=0))
        W =  np.sqrt(self.power/self.K) * W             
        for k in range(self.K):
            signal_strength = (np.linalg.norm(H[k,:] @ W[:,k]))**2
            # calculating interference from the other users
            interference = 0
            for j in range(self.K):
                if j != k:
                    interference += (np.linalg.norm(H[k,:] @ W[:,j]))**2
        
            rates.append(np.log2(1+(signal_strength/(noise + interference))))
        MRT_rate = np.sum(rates)
        return  MRT_rate
    
    def MRT_BF_Single_serving(self,H, W , u):
        
        noise = self.noise_var
        signal_strength = (np.linalg.norm(H[u,:] @  W[:,u]))**2
        snr = signal_strength/noise
        
        rate = np.log2(1 + snr)
        return rate
        
    
    # User Selection Algorithms:
    
    
    
    
    # Correlation based User selection:
    
    def Corr_US(self,heq,h1,h2,h3,X,NU_index):
        W_update = np.zeros((self.M,self.K), dtype='complex')

        #step 1 : Initialisation
        S_t = []
        W_up = heq[:,:]/np.linalg.norm((heq[:,:])**2,axis=0) 
        W_update = W_up.T
        # setting the PS of IRS to closest user.
        # phase0 = np.angle((h1[0,:]) @ W_update[:,0]) - np.angle((np.diag(h2[0,:]) @ h3[:,:] @ W_update[:,0]))
        # v = np.exp(1j*phase0)
        # Theta_corr = np.diag(v)
        
        
        phase_NU = np.angle((h1[NU_index,:]) @ W_update[:, NU_index])  -  np.angle((np.diag(h2[NU_index,:]) @ h3[:,:] @ W_update[:,NU_index]))
        v = np.exp(1j* phase_NU)
        Theta_corr = np.diag(v)
        
        
        #for the initialised phases [Theta], the overall channel is redefined:
        h = h1 + h2 @ Theta_corr @ h3
        
        sum1 = 0
        list_ratio = []
        norm = 0
        #step 2 : iterations to select  1st user
        H_s =[]  # store the channels for the selected users
        T1 = np.arange(self.K)
        h_list = []
        
        for t in range(X): 
            if t==0:
                for i in range(len(T1)):
                    h_list.append(h[i,:])
                h_norm = (np.linalg.norm(h, axis=1)) ** 2    # axis=1 means row wise norm calculation fo all columns
                #print("h_norm",h_norm)
                k1 = np.argmax(h_norm)
                S_t.append(k1)
                H_s.append(h[k1,:])
                T2 = [x for x in T1 if x != k1]
                #print(T2)

            
        # step 3 : Selecting more users.  
            else:
                #print("T2 after 1st user Selected=",T2)
                for i in T2:
                    c_ki = np.abs((np.conj(h[k1,:]).T) @ h[i,:])/((np.linalg.norm(h[k1,:]))*(np.linalg.norm(h[k1,:])))
                    #print("c_K_i :", c_ki)
                    sum1 = sum1 + c_ki**2
                    #print("sum:",sum1)
                    norm = np.linalg.norm(h[k1,:])**2
                    ratio =  norm/sum1
                    #print('ratio=',ratio)  
                    list_ratio.append(ratio)
                #print(list_ratio)
                k = np.argmax(list_ratio)
                k2 = T2[k]
                #print(k)
                #print("k2=",k2)
                S_t.append(k2)
                H_s.append(h[k2,:])
                T2 = [x for x in T2 if x != k2]
                #print("T2 after next user =",T2)
                    
                    
        return S_t, Theta_corr
    
    # ZF based User Selection
    def ZF_US( self,X, H1,H2,H3):  #H1 for Hd,   H2 for Hr &     H3 for G
        noise = 0
        #  step 1 : initialisation
        S_t = []   #to store the index of selected users
        W_update = np.zeros((self.M,self.K), dtype='complex')    # initialising the BF vectors

        # Initialising the phases at IRS
        phase0 = np.eye(self.N)
        v = np.exp(1j*phase0)
        Theta = np.diag(v.diagonal())
        #print(Theta.shape)
        
        #for the initialised phases [Theta], the overall channel is redefined:
        h = H1 + H2 @ Theta @ H3
        #print(h[3,:])
        #print(h[0,:])
        
        W2 = h/(np.linalg.norm(h))**2
        W2 = W2/np.linalg.norm(W2, axis=0)
        W2 = W2.T
        
        R_max = 0 
        T1 = np.arange(self.K)
        
        iter = X 
        g = np.zeros((self.K,self.M) ,dtype='complex')
        list_g =[]
        
        # reflected IRS-User channel 
        list_hr_norm = []
        for i in range(self.K):
            norm = (np.linalg.norm(H2[i,:]))**2
            list_hr_norm.append(norm)
        #list_hr_norm.sort(reverse=true)    
            
        # step 2: Selecting 1st user   
        for t in range(iter):
            if t==0:
                for i in range(len(T1)):
                    g[i,:] = h[i,:]
                    #list_g.append(g[i,:])
                h_norm = (np.linalg.norm(g, axis = 1)) ** 2
                #print(h_norm)
            
                k1 = np.argmax(h_norm)
                #print(k1)  
                S_t.append(k1)
                #print(S_t)
                #print(T1)

                # need to align the RIS to the user k
                phase_shift = np.angle(H1[k1,:] @ W2[:,k1]) - np.angle(H2[k1,:]) - np.angle(H3[:,:] @ W2[:,k1])
                v = np.exp(1j*phase_shift)
                Theta1 = np.diag(v)
                
                R_max = (np.log2( 1 + ((np.linalg.norm(h[k1,:] @ W2[:,k1])) **2)/noise))
                T2 = [x for x in T1 if x != k1]
                #print(T2)
                
            # step 3: Iterative Greedy User Selection
            
            else:
                
                sum_rate = 0
                for i in range(len(T2)):
                    
                    H_comp =  np.array((h[k1,:],h[i,:]))
                    #print(H_comp.shape)
                    W_update = np.linalg.pinv(H_comp)
                    W_update = W_update/np.linalg.norm(W_update, axis=0)
                    #print(W_update.shape)
                    
                    rates_list = []
                    
                    #rate1 = (np.log2( 1 + ((np.linalg.norm(H_comp[0,:] @ W_update[:,0])) **2)/noise))
                    #print(rate1)
                    rate2 = (np.log2( 1 + ((np.linalg.norm(H_comp[1,:] @ W_update[:,1])) **2)/noise)) + (np.log2( 1 + ((np.linalg.norm(H_comp[0,:] @ W_update[:,0])) **2)/noise))
                    #print(rate2)
                    rates_list.append(R_max)
                    rates_list.append(rate2)
                    #print(rates_list)
                    
                    # now comparing the rate
                    if rate2 > R_max:
                        k=i
                        S_t.append(k)
                        #print(S_t)
                        #print()
                        
                        T2 = [x for x in T2 if x != k]
                        if list_hr_norm[k] > list_hr_norm[k1]:
                            phase_shift = np.angle(H1[k,:] @ W2[:,k]) - np.angle(H2[k,:]) - np.angle(H3[:,:] @ W2[:,k])
                            v = np.exp(1j*phase_shift)
                            Theta1 = np.diag(v)
                        R_max = R_max + rate2
                            
                    # else:
                    #     break
                        
    
        return S_t, Theta1 
    
    def Rate_User_Selection(self,H, S2):  # S2 is the selected users list
        rates1 = []
        rates2 = []
        
        noise = self.noise_var
        #print("H", H)
        # MRT Beamforming:
        W1 = H.conj().T
        W1 = W1/(np.linalg.norm(W1,axis=0))
        W1 =  np.sqrt(self.power/self.X) * W1 #(np.conj(H).T)/ np.linalg.norm(H)
        
        # ZF Beamforming:
        W2 = np.linalg.pinv(H)
        #W2 = W2/(np.linalg.norm(W2, axis=0))
        #W2 = np.sqrt(self.power/self.X) * W2
        W2 = W2/(np.linalg.norm(W2))
        W2 = np.sqrt(self.power) * W2
        
        #print("X",self.X)
        #print(H.shape)  # 2x4
        #print(W2.shape) # 4x2
        
        for k in range(len(S2)):
            signal_strength = (np.abs(H[k,:] @ W1[:,k]))**2
            # calculating interference from the other users
            interference = 0
            for j in range(len(S2)):
                if j != k:
                    interference += (np.abs(H[k,:] @ W1[:,j]))**2
                    #print("MRT interference", interference)
                    #print("MRT SS", signal_strength)
            rates1.append(np.log2(1+(signal_strength/(noise + interference))))
        MRT_rate = np.sum(rates1)
        #print("int MRT", interference)
                
        for k in range(len(S2)):
            signal_strength = (np.abs(H[k,:] @ W2[:,k]))**2
            interference = 0
            for j in range(len(S2)):
                if j != k:
                    interference += (np.abs(H[k,:] @ W2[:,j]))**2
                    #print("ZF interference", interference)
                    #print("ZF SS", signal_strength)                
            rates2.append(np.log2(1+(signal_strength/(noise + interference))))
        ZF_rate = np.sum(rates2)
        #print("MRT",MRT_rate,"ZF", ZF_rate,interference)
        
        return MRT_rate, ZF_rate   


    def Rate_User_Selection2(self,H, S2):  # S2 is the selected users list
        
        rates2 = []
        noise = self.noise_var
        
        # # ZF Beamforming:
        W2 = np.linalg.pinv(H)
        #print(W2.shape)
        W2 = W2/(np.linalg.norm(W2, axis=0))
        W2 = np.sqrt(self.power/len(S2)) * W2
                
        for k in range(len(S2)):
            signal_strength = (np.abs(H[k,:] @ W2[:,k]))**2
            interference = 0
            for j in range(len(S2)):
                if j != k:
                    interference += (np.abs(H[k,:] @ W2[:,j]))**2
                    #print("ZF interference", interference)
                    #print("ZF SS", signal_strength)                
            rates2.append(np.log2(1+(signal_strength/(noise + interference))))
        ZF_rate = np.sum(rates2)
       
        W = np.zeros((self.M,self.K),dtype=complex)
        W[:,S2] = W2
        #print("W2:",W2)
        #print("W:",W)
        #print("W size:",W.shape)

        return  ZF_rate,W             
    
    def Rate_US_MMSE(self,H,S2):
        rates_mmse = []
        noise = self.noise_var
        # Calculating the BF Vector
        W1 = (H @ np.conj(H).T) + noise * np.eye(len(S2))
        W2 = np.linalg.inv(W1)
        W_mmse = np.conj(H).T  @ W2
        W_mmse = W_mmse/(np.linalg.norm(W_mmse, axis=0))
        W_mmse = np.sqrt(self.power/len(S2)) * W_mmse

        # finding the sum rate:
        for k in range(len(S2)):
            signal_strength = (np.abs(H[k,:] @ W_mmse[:,k]))**2
            interference = 0
            for j in range(len(S2)):
                if j != k:
                    interference += (np.abs(H[k,:] @ W_mmse[:,j]))**2
                    #print("ZF interference", interference)
                    #print("ZF SS", signal_strength)                
            rates_mmse.append(np.log2(1+(signal_strength/(noise + interference))))
        mmse_rate = np.sum(rates_mmse)
       
        W = np.zeros((self.M,self.K),dtype=complex)
        W[:,S2] = W_mmse
        return  mmse_rate,W 
    # Semi Orthogonal User selection
    
    def SUS_Algo(self, H, alpha):
     
        H_new = np.zeros((self.X,self.M), dtype=complex)
        G_new = np.zeros((self.X,self.M), dtype=complex)
        alpha =1
        #step 1
        T1 = (np.arange(self.K))
        # T1 = [0,1,2,3]
        
        S0 = []
        Gk = np.zeros((self.K,self.M),dtype=complex)
        sum_k = 0

        # selecting 1st user
        Gk = H 
        norm_Gk = np.linalg.norm(Gk,axis=1) #4x1 column vector
        norm_Gk = (norm_Gk).T
        pi_1 = np.argmax(norm_Gk)  # anything from {0,1,2,3}
        #print("pi(1)=",pi_1)
        S0.append(pi_1)
        #print("1st selected user=",T1[index])
        h_1 = H[pi_1,:]
        g_1 = Gk[pi_1,:]

        H_new[0,:] = h_1
        G_new[0,:] = g_1

        # since alpha=1, all users other than pi_1 will be orthogonal to g_1.
        T_2 =[x for x in T1 if x != pi_1]
        #print("T_2= ",T_2)

        g_list = []
        for k in T_2:
            g_k = H[k,:] -  ((H[k,:] @ np.conj(g_1).T)/np.linalg.norm(g_1)**2) * g_1
            g_k_norm = np.linalg.norm(g_k)
            g_list.append(g_k_norm)

        sel_in = np.argmax(g_list) # index value={0,1,2}
        pi_2 = T_2[sel_in]
        #print("pi(2)=",pi_2)
        S0.append(pi_2)
        h_2 = H[pi_2,:]
        g_2 = Gk[pi_2,:]

        H_new[1,:] = h_2
        G_new[1,:] = g_2

        # since alpha=1, all users other than pi_2 will be orthogonal to g_1 & g_2.
        T_3 =[x for x in T_2 if x != pi_2]
        #print("T_3= ",T_3)

        if len(S0)<self.X:
            g_list = []
            for k in T_3:
                g_k = H[k,:] -  ((H[k,:] @ np.conj(g_1).T)/np.linalg.norm(g_1)**2) * g_1  - ((H[k,:] @ np.conj(g_2).T)/np.linalg.norm(g_2)**2) * g_2
                g_k_norm = np.linalg.norm(g_k)
                g_list.append(g_k_norm)
            
            sel_in = np.argmax(g_list)
            pi_3 = T_3[sel_in]
            S0.append(pi_3)
        S0 = np.array(S0)
        return S0


            
# mean_sr_s1_seminar =

#     2.1915
#     2.8860
#     3.6943
#     4.6534
#     5.8009
#     7.1481
#     8.7008
#    10.4571
#    12.3968
#    14.4973
#    16.7324
#    19.0752
#    21.5030
#    23.9933
#    26.5335
#    29.1134
#    31.7150
#    34.3343
#    36.9671
#    39.6083
#    42.2564
#    44.9058
#    47.5595
#    50.2124
#    52.8682
#    55.5238

# >> 

# mean_sr_s2 =

#     0.3590
#     0.5435
#     0.8067
#     1.1687
#     1.6459
#     2.2493
#     2.9771
#     3.8168
#     4.7235
#     5.6848
#     6.7698
#     8.0297
#     9.4766
#    11.1157
#    12.9451
#    14.9468
#    17.0941
#    19.3639
#    21.7299
#    24.1768
#    26.6864
#    29.2417
#    31.8295
#    34.4401
#    37.0646
#    39.6995



