# All Users grouped togethewr near RIS.

import numpy as np
import random

class Channel_Generation_6:
    
    def __init__(self, K, M, N):
        self.K = K
        self.M = M
        self.N = N
        self.epsilon_G = 10
        self.epsilon_Hr = 10
        
    def calc_distance(self, loc1, loc2):
        distance = np.sqrt((loc1[0]-loc2[0])**2  + (loc1[1]-loc2[1])**2   +  (loc1[2]-loc2[2])**2)
        return distance
    
    
    def User_locations(self, x_min,y_min,  x_max,y_max):
        count= self.K
        User_loc = []
        for _ in range(count):
            x = random.uniform(x_min,x_max)
            y = random.uniform(y_min,y_max)
            z=0
            User_loc.append((x,y,z))
        User_loc = [(15,15,0), (10,20,0), (15,10,0), (20,20,0)]
       
        return User_loc
        
    def Channels(self):
        
        # initialise the path loss parameters. case 2 LoS path is non dominant.
        alpha_AI = 2
        alpha_Au = 3.8
        alpha_Iu = 3.5
        
        #Initialising the channels.
        Hd = np.zeros((self.M, self.K), dtype= complex)
        Hr = np.zeros((self.N, self.K), dtype= complex)
        
        # AP and RIS locations
        RIS = (0,0,0)  # at origin
        BS = (100,0,0) # XZ plane
        
        # User locations
        User = self.User_locations(10,10,  20,20)
        #print("User locations=", User)
        
        # BS -- RIS distance
        dist_AI = self.calc_distance(BS, RIS)
        
        # Steering vector from BS to RIS:
        SV_BS = np.exp(1j* np.pi *np.array(range(self.M)) * ((RIS[0]-BS[0])))/dist_AI   
        SV_RIS = np.exp(1j*np.pi * ((np.mod(range(self.N), 10) * (BS[1]-RIS[1])/dist_AI) + (np.floor(np.array(range(self.N))/10) * (BS[2]-RIS[2])/dist_AI)))
        
        
        ########################################################################
        # Steering vector from RIS to Users:
        # for k in range(self.K):
        #     dist_Iu = self.calc_distance(RIS,  User[k])
        #     #SV_RIS = np.exp(1j*np.pi * ((np.mod(range(self.N), 10) * (User[k][1]-RIS[1])/dist_Iu) + (np.floor(np.array(range(self.N))/10) * (User[k][2]-RIS[2])/dist_Iu)))
        ########################################################################
        
        
            
        # Channel from BS to RIS:
        beta_AI = 10**(-3) * dist_AI**(-alpha_AI)
        
        G_NLoS = (np.random.randn(self.M, self.N) - 1j * np.random.randn(self.M, self.N)) / np.sqrt(2)
        G_LoS = SV_BS.reshape(-1, 1) @ SV_RIS.reshape(1, -1)
        
        G= np.sqrt(beta_AI)*(np.sqrt(self.epsilon_G/(1 + self.epsilon_G)) * G_LoS + np.sqrt(1/(1 + self.epsilon_G)) * G_NLoS)
        G = G.T 
        
        # Finding the nearest user from RIS located at (0,0,0):
        distance_list = []
        for k in range(self.K):
            self.dist = self.calc_distance((0,0,0), User[k])
            distance_list.append(self.dist)
        nearest_user = np.argmin(distance_list)

        
        # Channel from BS to User:
        for k in range(self.K):
            dist_Au = self.calc_distance(BS, User[k])
            dist_Iu = self.calc_distance(RIS, User[k])
            
            beta_Au= 10**(-3) * dist_Au**(-alpha_Au) *10**(-3)   #changed from -3 to -4.5                                                                                                                                             
            beta_Iu= 10**(-3) * dist_Iu**(-alpha_Iu)

            Hd[:, k]= np.sqrt(beta_Au) * (np.random.randn(self.M) + 1j * np.random.randn(self.M)) / np.sqrt(2)
            #steering_vector_Iu= np.exp(1j*np.pi * ((np.mod(range(self.N), 10) * (loc_users[k][1]-loc_IRS[1])/d_Iu) + (np.floor(np.array(range(self.N))/10) * (loc_users[k][2]-loc_IRS[2])/d_Iu)))
            SV_Users= np.exp(1j*np.pi * ((np.mod(range(self.N), 10) * (User[k][1]-RIS[1])/dist_Iu) + (np.floor(np.array(range(self.N))/10) * (User[k][2]-RIS[2])/dist_Iu)))
            hr_LoS=SV_Users
            hr_NLoS= (np.random.randn(self.N) + 1j * np.random.randn(self.N)) / np.sqrt(2)

            Hr[:, k]= np.sqrt(beta_Iu) * ((np.sqrt(self.epsilon_Hr/(1 + self.epsilon_Hr)) * hr_LoS) + (np.sqrt(1/(1 + self.epsilon_Hr)) * hr_NLoS))
        return Hd, Hr, G, nearest_user
    
    # def Nearest_User(self,User):
    #     #User2 = self.User_locations(3,30,  10,40)

    #     distance_list = []
    #     for k in range(self.K):
    #         self.dist = self.calc_distance((0,0,0), User[k])
    #         distance_list.append(self.dist)
    #     user_index = np.argmin(distance_list)
    #     return user_index