# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 03:19:27 2023

@author: Fedosov
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy

import torch


plt.close('all')

np.random.seed(0)




class Batch2d():
    def __init__(self, C,WH, momentum = 0.0003):
        self.mean = torch.zeros([1,C,WH,WH])
        self.var = torch.ones([1,C,WH,WH])
        self.moment = momentum
    def apply(self,x):
        x = x.detach().clone()
        self.mean = (1.0-self.moment)*self.mean+x*self.moment
        self.var =  (1.0-self.moment)*self.var+(x-self.mean)*(x-self.mean)*self.moment
        #return (x-self.mean)/torch.sqrt(self.var)
  
class Batch0d():
    def __init__(self, C, momentum = 0.0003):
        self.mean = torch.zeros([1,C])
        self.var = torch.ones([1,C])
        self.moment = momentum
    def apply(self,x):
        x = x.detach().clone()
        self.mean = (1.0-self.moment)*self.mean+x*self.moment
        self.var =  (1.0-self.moment)*self.var+(x-self.mean)*(x-self.mean)*self.moment
        #return (x-self.mean)/torch.sqrt(self.var)



class AI(torch.nn.Module):
  
    def __init__(self):
        super().__init__()
        self.N_states = 10
        
        
        
        
        self.post_conv0 = torch.nn.Conv2d(in_channels=3,out_channels = 40,kernel_size = [3,3],padding = 1)#,stride = [2,2])
        #self.post_batch0 = Batch2d(40,100,  momentum=0.0003)
        
        self.post_conv1 = torch.nn.Conv2d(in_channels=40,out_channels = 80,kernel_size = [5,5],padding = 2)#,stride = [2,2])
        self.post_max1 = torch.nn.MaxPool2d(4,return_indices = True)

        
        self.post_conv2 = torch.nn.Conv2d(in_channels=80,out_channels = 160,kernel_size = [5,5],padding = 2)#,stride = [2,2])
        self.post_max2 = torch.nn.MaxPool2d(4,return_indices = True)
        
        
        
        self.post_fc5 = torch.nn.Linear(in_features = 6*6*160+400,out_features = 800)

    
        self.post_fc6_mu = torch.nn.Linear(in_features = 800,out_features = 100)
        self.post_fc6_var = torch.nn.Linear(in_features = 800,out_features = 100)
        
        
        self.post_lstm_fc0 = torch.nn.Linear(in_features = 102,out_features = 400)
        #self.trans_fc1 = torch.nn.Linear(in_features = 103,out_features = 103)
        self.post_lstm = torch.nn.LSTMCell(400, 400)
    
          
        
        self.like_fc0 = torch.nn.Linear(in_features = 100,out_features = 800)
        self.like_fc1 = torch.nn.Linear(in_features = 800,out_features = 6*6*160)
        
        self.like_upsample1 = torch.nn.Upsample(size = [25,25])
 
        self.like_decon2 = torch.nn.ConvTranspose2d(in_channels=160,out_channels = 80,kernel_size = [5,5],padding = 2)#,stride = [2,2])
        self.like_upsample2 = torch.nn.Upsample(size = [100,100])
        
   

        self.like_decon5 = torch.nn.ConvTranspose2d(in_channels=80,out_channels = 40,kernel_size = [5,5],padding = 2)#,stride = [2,2])
        self.like_decon6 = torch.nn.Conv2d(in_channels=40,out_channels = 3,kernel_size = [3,3],padding = 1)#,stride = [2,2])
        


        
        self.value_fc0 = torch.nn.Linear(in_features = 100,out_features = 100)
        self.value_fc1_ = torch.nn.Linear(in_features = 100,out_features = 50)
        self.value_fc2_ = torch.nn.Linear(in_features = 50,out_features = 1)
        #self.value_fc2 = torch.nn.Linear(in_features = 80,out_features = 1)
        
        
        "state space = 100 "
        
        self.habit_fc0_ = torch.nn.Linear(in_features = 102,out_features = 200)
        self.habit_fc1_ = torch.nn.Linear(in_features = 200,out_features = 200)
        self.habit_fc2_ = torch.nn.Linear(in_features = 200,out_features = 2)
        
        self.recon_fc0 = torch.nn.Linear(in_features = 102,out_features = 200)
        self.recon_fc1 = torch.nn.Linear(in_features = 200,out_features = 200)
        self.recon_fc2 = torch.nn.Linear(in_features = 200,out_features = 2)
        
        
        self.critic_fc0 = torch.nn.Linear(in_features = 102,out_features = 200)
        self.critic_fc1_sig = torch.nn.Linear(in_features = 200,out_features = 100)
        self.critic_fc1_rel = torch.nn.Linear(in_features = 200,out_features = 100)
        self.critic_fc2_ = torch.nn.Linear(in_features = 100,out_features = 20)
        
        
        
        "motor space = 2, state space = 50, directly observed state - 1 "
        
        self.trans_fc0 = torch.nn.Linear(in_features = 102,out_features = 400)
        #self.trans_fc1 = torch.nn.Linear(in_features = 103,out_features = 103)
        self.trans_lstm = torch.nn.LSTMCell(400, 400)
        
        self.trans_fc1_mu = torch.nn.Linear(in_features = 400,out_features = 100)
        self.trans_fc1_var = torch.nn.Linear(in_features = 400,out_features = 100)
        
        #self.avg_pool1 = torch.nn.AvgPool2d(kernel_size = 5)
        #self.post_conv2 = torch.nn.Conv2d(in_channels=16,out_channels = 32,kernel_size = [7,7],stride = [2,2])
        
        #self.post_obs_fc0 = torch.nn.Linear(self.N_obs,self.N_obs)
        
        
        self.world = World()
        
        

        
        



    #Dkl(p||q) - q - prior, p - posterior for VAE
    def GDKLoss(self,mu_p, logvar_p, mu_q, logvar_q):
        """
        Compute the KL divergence between two Gaussian distributions with diagonal covariance matrices.
        mu_q: mean of the first Gaussian distribution
        logvar_q: log-variance of the first Gaussian distribution
        mu_p: mean of the second Gaussian distribution
        logvar_p: log-variance of the second Gaussian distribution
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (logvar_q - logvar_p + var_p / var_q + ((mu_q - mu_p)**2) / var_q - 1)
 
        return kl.mean()
    
    
    
    def BLike(self,tar_s,predicted_pb):
        #tar_s[tar_s<0.01] = torch.tensor(0.01)
        #tar_s[tar_s>0.99] =  torch.tensor(0.99)             
        L = -torch.mean(tar_s*torch.log(predicted_pb)+(1.0-tar_s)*torch.log(1.0-predicted_pb))
        return L
    
    def GLike(self,tar_s, predicted_pb):
        L = 0.5*torch.mean((tar_s-predicted_pb)**2)      
        return L
    

    def plot_torch_obs(self, image):
        image= image[0,:,:,:].detach().numpy().transpose((1,2,0))
        image = np.clip(image, 0.0, 1.0)
        
        plt.figure()
        plt.imshow(image.transpose((1,0,2))[:,::-1,:], origin = 'lower')
             
 
        
    #сначала попробуем обучать как VAE
    def imag_train_habit(self):
        N_rollouts = 100
        N_depth = 50
        #N_branches = 100
  
        loss_history = np.zeros(N_rollouts)
        
        
        
        # train the critic network
        # and parallely - habit
        
        for i in range(N_rollouts):
            pass
         
        
 
    def init_actionet(self):

        
        #first, we train actionet with just entropy loss
        
        #then, in second stage, we train G based on it
        
        #after that, the main cycle goes
        
        #self.action_optimizer = torch.optim.Adam(self.parameters()??, lr=0.0003)
        
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay = 0.1)
        self.action_optimizer = torch.optim.Adam([
    {'params': self.habit_fc0_.parameters()},
    {'params': self.habit_fc1_.parameters()},
    {'params': self.habit_fc2_.parameters()}, 
    {'params': self.recon_fc0.parameters()},                  # First layer without regularization
     {'params': self.recon_fc1.parameters()},
     {'params': self.recon_fc2.parameters()},
    
    ], lr=0.001)
        
        
        N_acts_init = 1000
        
        
        N_step = 10
        N_trajectories = 20
        
        # ten different world start
        #N_rollouts_per_batch = 10
        
        
        loss_history = np.zeros(N_acts_init)
        
        for i in range(N_acts_init):
            
            self.action_optimizer.zero_grad()
            
            cumulative_loss = 0.0
            

            bias = torch.from_numpy((np.array([0x6a,0xb8,0x28])/255.0).astype('float32'))[None,:,None,None]#np.random.rand(self.N_vis,self.N_vis)/10+0.7
            
            for j in range(N_trajectories):
            

                self.world.generate_world()
                self.world.init_random_pos()
            
            
                self.world.render_image(plot = False) 
                self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                self.value = torch.tensor([self.world.value])[:,None]
            
            
                self.state_sample = torch.zeros([1,100])
                self.action_sample= torch.zeros([1,2])
                self.h_post_lstm = torch.zeros([1,400])
                self.c_post_lstm = torch.zeros([1,400])
            
            
                self.posterior()
                self.likelihood() # NOT NECESSARY
                self.valuenet() # NOT NECESSARY
            
            
            
                self.actionet(1)
                self.action_entropy()
            
                self.critic()
            
                #0.0001 is too high!!!!
                loss = torch.mean(self.predict_G)*0.0+self.GLike(self.intention_noise.detach().clone(),self.intention_recon)#+torch.mean(p*torch.log(p))#-torch.mean(T)#torch.mean(-torch.log(x))#self.GLike(self.target,x)*0.1\
            
                loss /= (N_trajectories*(N_step-1))
                loss.backward()
                
                cumulative_loss += loss.detach().numpy()
            
                self.h_lstm = torch.zeros([1,400])
                self.c_lstm = torch.zeros([1,400])
            
            
                dist_coord = np.random.rand()
                orient_coord = np.random.rand()*2-1.0  
                
                self.action_sample = torch.Tensor([[dist_coord],[orient_coord]]).T   
                
            
                for k in range(N_step):
                
                
                
                    self.transition()
                    self.state_sample = self.state_trans_sample.detach()
                    
                    self.likelihood() #not necessary
                    self.valuenet() # not necessary, for visualization
                    
                    #self.state_sample = self.state_sample.detach()
                    
                    
                    self.actionet(1)
                    self.action_entropy()
                
                    self.critic()
                    
                    loss = torch.mean(self.predict_G)*0.0+self.GLike(self.intention_noise.detach().clone(),self.intention_recon)#+torch.mean(p*torch.log(p))#-torch.mean(T)#torch.mean(-torch.log(x))#self.GLike(self.target,x)*0.1\
                
                    loss /= (N_trajectories*(N_step-1))
                    loss.backward()
                    
                    cumulative_loss += loss.detach().numpy()
                
    
                    dist_coord = np.random.rand()
                    orient_coord = np.random.rand()*2-1.0  
                    
                    self.action_sample = torch.Tensor([[dist_coord],[orient_coord]]).T 
                    
                #print(j)
                    
            self.action_optimizer.step()
            loss_history[i] = cumulative_loss
            print(i, loss_history[i])
            
        plt.figure()
        plt.plot(loss_history)
            
                    
     
            
      
    def plot_actiohist(self, N_trajectory = 1, N_samples = 10000):
        
        bias = torch.from_numpy((np.array([0x6a,0xb8,0x28])/255.0).astype('float32'))[None,:,None,None]#np.random.rand(self.N_vis,self.N_vis)/10+0.7
        
        
        N_step = 10
        # for the random trajectory plot hist for every step
        for i in range(N_trajectory):
        

            self.world.generate_world()
            self.world.init_random_pos()
        
        
            self.world.render_image(plot = False) 
            self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
            self.value = torch.tensor([self.world.value])[:,None]
        
        
            self.state_sample = torch.zeros([1,100])
            self.action_sample= torch.zeros([1,2])
            self.h_post_lstm = torch.zeros([1,400])
            self.c_post_lstm = torch.zeros([1,400])
        
        
            self.posterior()
            self.likelihood() # NOT NECESSARY
            self.valuenet() # NOT NECESSARY
        
        
        
            self.actionet(N_samples)
            
            plt.figure()
            plt.hist2d(self.action_sample[:,0].detach().numpy(), 
                                     self.action_sample[:,1].detach().numpy(), bins=20, 
                                     range = [[0, 1], [-1, 1]])
            
           
           
            self.h_lstm = torch.zeros([1,400])
            self.c_lstm = torch.zeros([1,400])
        
        
            dist_coord = np.random.rand()
            orient_coord = np.random.rand()*2-1.0  
            
            self.action_sample = torch.Tensor([[dist_coord],[orient_coord]]).T   
            
        
            for k in range(N_step):
            
            
            
                self.transition()
                self.state_sample = self.state_trans_sample
                self.likelihood() #not necessary
                self.valuenet() # not necessary, for visualization
                
                
                self.actionet(N_samples)
                
                plt.figure()
                plt.hist2d(self.action_sample[:,0].detach().numpy(), 
                                         self.action_sample[:,1].detach().numpy(), bins=20, 
                                         range = [[0, 1], [-1, 1]])
                plt.colorbar()
                
                self.Gplt()
                
                
                dist_coord = np.random.rand()
                orient_coord = np.random.rand()*2-1.0  
                
                self.action_sample = torch.Tensor([[dist_coord],[orient_coord]]).T 
             
                
             
    
                
           
    
    def init_experiment(self, train_EFE = True):

        
        #first, we train actionet with just entropy loss
        
        #then, in second stage, we train G based on it
        
        #after that, the main cycle goes
        
        #self.action_optimizer = torch.optim.Adam(self.parameters()??, lr=0.0003)
        
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay = 0.1)
        self.critic_optimizer = torch.optim.Adam([
    {'params': self.critic_fc0.parameters()},
    {'params': self.critic_fc1_sig.parameters()},
    {'params': self.critic_fc1_rel.parameters()}, 
    {'params': self.critic_fc2_.parameters()},                  # First layer without regularization
    ], lr=0.001)
        
        self.action_optimizer = torch.optim.Adam([
    {'params': self.habit_fc0_.parameters()},
    {'params': self.habit_fc1_.parameters()},
    {'params': self.habit_fc2_.parameters()}, 
    {'params': self.recon_fc0.parameters()},                  # First layer without regularization
     {'params': self.recon_fc1.parameters()},
     {'params': self.recon_fc2.parameters()},
    
    ], lr=0.001) #slower learning
        
        
        N_acts_init = 250 #1000
        
        
        N_step = 5
        N_trajectories = 200
        
        N_samples = 50 # for the mean entropy, reward, and G+1 calculation
        
        # ten different world start
        #N_rollouts_per_batch = 10
        
        
        loss_history = np.zeros(N_acts_init)
        loss_UTIL = np.zeros(N_acts_init)
        loss_H = np.zeros(N_acts_init)
        
        for i in range(N_acts_init):
            
            if train_EFE:
                self.critic_optimizer.zero_grad()
                self.action_optimizer.zero_grad()
                
                cumulative_loss = 0.0
     
                bias = torch.from_numpy((np.array([0x6a,0xb8,0x28])/255.0).astype('float32'))[None,:,None,None]#np.random.rand(self.N_vis,self.N_vis)/10+0.7
                
                for j in range(N_trajectories):
                
    
                    self.world.generate_world()
                    self.world.init_random_pos()
                
                
                    self.world.render_image(plot = False) 
                    self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                    self.value = torch.tensor([self.world.value])[:,None]
                
                
                    self.state_sample = torch.zeros([1,100])
                    self.action_sample= torch.zeros([1,2])
                    self.h_post_lstm = torch.zeros([1,400])  #STOP
                    self.c_post_lstm = torch.zeros([1,400])
                
                    
                
                
                    #??? Here some additional sampling
                    self.posterior()
                    
                    
                    
    
                    self.h_lstm = torch.zeros([1,400])
                    self.c_lstm = torch.zeros([1,400])
                
                
                    self.actionet()
                  
                    
                    for k in range(N_step):
                        
                        
                        
                        self.critic()
                        
                        self.action_sample = self.action_sample.detach().clone()
                        self.state_sample = self.state_sample.detach().clone()
                        self.h_lstm = self.h_lstm.detach().clone()
                        self.c_lstm = self.c_lstm.detach().clone()
                        
                        
                        # HERE I SHOULD COMPUTE THE action loss
                        
                        
                        predicted_critic = self.predict_G.clone()
                
                    
                        self.transition(N_samples = N_samples)
                        self.state_sample = self.state_trans_sample.detach().clone()
                        
                        
                        remember_one_state = self.state_sample[[0],:].clone()
                        #remember the state to reassign and take action
                        
                        next_step_entropy =(0.5+0.5*(np.log(2*np.pi)+self.state_logvar)).detach().clone()
                        
                        
                        
                        #self.likelihood() # MULTIPLE!!!!!!!! AND THEN BACK AGAIN!!!!
                        self.valuenet() 
                        #self.obs = self.obs_pb.clone()
                        
                      
                        
                        r = torch.mean(self.predict_value.detach().clone(),dim = 0)  #NOT AS IN THEORY!!!
                        
                          
                        uncertainty_entropy = 0#(0.5+0.5*(np.log(2*np.pi)+torch.mean(self.state_logvar,dim = 0))).detach().clone()
                    
                        self.actionet()
                        self.critic()
                        
                        decay_coef = 0.8
                        boots_G = torch.mean(self.predict_G,dim = 0)
    
                        #-0.0*(next_step_entropy-uncertainty_entropy)
                        G_est = -r+boots_G*decay_coef
                    
                        loss = self.GLike(G_est.detach().clone(),predicted_critic)*100
                        
                        loss /= (N_trajectories*(N_step-1))
                        
                        #self.turn_onoff_critic_tr(True)
                        loss.backward()
                        #self.turn_onoff_critic_tr(False)
                        
                        
                        self.state_sample = remember_one_state.detach().clone()
                      
                        self.actionet()
                    
                        cumulative_loss += loss.detach().numpy()
                        
                        
                        
                self.critic_optimizer.step()
           
                loss_history[i] = cumulative_loss
     
                
                print(i, loss_history[i])
                
            
            
            #####################(((())))#################################
            self.critic_optimizer.zero_grad()
            self.action_optimizer.zero_grad()
            
           
            cumulative_loss_UTIL = 0.0
            cumulative_loss_H = 0.0

            bias = torch.from_numpy((np.array([0x6a,0xb8,0x28])/255.0).astype('float32'))[None,:,None,None]#np.random.rand(self.N_vis,self.N_vis)/10+0.7
            
            for j in range(N_trajectories):
            

                self.world.generate_world()
                self.world.init_random_pos()
            
            
                self.world.render_image(plot = False) 
                self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                self.value = torch.tensor([self.world.value])[:,None]
            
            
                self.state_sample = torch.zeros([1,100])
                self.action_sample= torch.zeros([1,2])
                self.h_post_lstm = torch.zeros([1,400])  #STOP
                self.c_post_lstm = torch.zeros([1,400])
            
                
            
            
                #??? Here some additional sampling
                self.posterior()
                
                
                

                self.h_lstm = torch.zeros([1,400])
                self.c_lstm = torch.zeros([1,400])
            
            
                self.actionet()
                self.action_entropy()
                
                
            
                for k in range(N_step):
                    
                    
                    self.critic()
                    #0.0001 too high
                    loss_action = torch.mean(self.predict_G)*0.7+self.GLike(self.intention_noise.detach().clone(),self.intention_recon)*1.0#+torch.mean(p*torch.log(p))#-torch.mean(T)#torch.mean(-torch.log(x))#self.GLike(self.target,x)*0.1\
                    loss_action /= (N_trajectories*(N_step-1))
                    
                    #self.turn_onoff_actionet_tr(True)
                    loss_action.backward(retain_graph = True)
                    #self.turn_onoff_actionet_tr(False)
                    
                    cumulative_loss_UTIL += torch.mean(self.predict_G).detach().numpy()*0.3/(N_trajectories*(N_step-1))
                    cumulative_loss_H += (self.GLike(self.intention_noise.detach().clone(),self.intention_recon)).detach().numpy()/\
                        (N_trajectories*(N_step-1))

                
                    
                    self.action_sample = self.action_sample.detach().clone()
                    self.state_sample = self.state_sample.detach().clone()
                    self.h_lstm = self.h_lstm.detach().clone()
                    self.c_lstm = self.c_lstm.detach().clone()
                    
                    
                    # HERE I SHOULD COMPUTE THE action loss
                    
                    
            
                
                    self.transition()
                    self.state_sample = self.state_trans_sample.detach().clone()
                    
                   
                    self.actionet()
                    self.action_entropy()
                    
                   
                 
     
            self.action_optimizer.step()
    
            loss_UTIL[i] = cumulative_loss_UTIL
            loss_H[i] = cumulative_loss_H
            
            print(i, loss_UTIL[i], loss_H[i])
        
            
        plt.figure()
        plt.plot(loss_history)
        plt.figure()
        plt.plot(loss_UTIL)
        plt.figure()
        plt.plot(loss_H)
        
        
        
        
        
        
        
    def turn_onoff_valuenet_tr(self,cond):
        self.value_fc0.weight.requires_grad = cond
        self.value_fc0.bias.requires_grad = cond
        
        self.value_fc1_.weight.requires_grad = cond
        self.value_fc1_.bias.requires_grad = cond

        self.value_fc2_.weight.requires_grad = cond
        self.value_fc2_.bias.requires_grad = cond
               
                    
    def turn_onoff_critic_tr(self,cond):
        self.critic_fc0.weight.requires_grad = cond
        self.critic_fc0.bias.requires_grad = cond
        
        self.critic_fc1_rel.weight.requires_grad = cond
        self.critic_fc1_rel.bias.requires_grad = cond
        self.critic_fc1_sig.weight.requires_grad = cond
        self.critic_fc1_sig.bias.requires_grad = cond

        self.critic_fc2_.weight.requires_grad = cond
        self.critic_fc2_.bias.requires_grad = cond
        
        

    def turn_onoff_actionet_tr(self,cond):
        self.habit_fc0_.weight.requires_grad = cond
        self.habit_fc0_.bias.requires_grad = cond
        
        self.habit_fc1_.weight.requires_grad = cond
        self.habit_fc1_.bias.requires_grad = cond

        self.habit_fc2_.weight.requires_grad = cond
        self.habit_fc2_.bias.requires_grad = cond
        
        
        self.recon_fc0.weight.requires_grad = cond
        self.recon_fc0.bias.requires_grad = cond
        
        self.recon_fc1.weight.requires_grad = cond
        self.recon_fc1.bias.requires_grad = cond

        self.recon_fc2.weight.requires_grad = cond
        self.recon_fc2.bias.requires_grad = cond
        
    
   
                
     
    
    
    def init_critic(self):

        
        #first, we train actionet with just entropy loss
        
        #then, in second stage, we train G based on it
        
        #after that, the main cycle goes
        
        #self.action_optimizer = torch.optim.Adam(self.parameters()??, lr=0.0003)
        
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay = 0.1)
        self.critic_optimizer = torch.optim.Adam([
    {'params': self.critic_fc0.parameters()},
    {'params': self.critic_fc1_sig.parameters()},
    {'params': self.critic_fc1_rel.parameters()}, 
    {'params': self.critic_fc2_.parameters()},                  # First layer without regularization
    ], lr=0.0003)
        
        
        N_acts_init = 1000
        
        
        N_step = 10
        N_trajectories = 200
        
        N_samples = 50 # for the mean entropy, reward, and G+1 calculation
        
        # ten different world start
        #N_rollouts_per_batch = 10
        
        
        loss_history = np.zeros(N_acts_init)
        
        for i in range(N_acts_init):
            
            self.critic_optimizer.zero_grad()
            
            cumulative_loss = 0.0
            

            bias = torch.from_numpy((np.array([0x6a,0xb8,0x28])/255.0).astype('float32'))[None,:,None,None]#np.random.rand(self.N_vis,self.N_vis)/10+0.7
            
            for j in range(N_trajectories):
            

                self.world.generate_world()
                self.world.init_random_pos()
            
            
                self.world.render_image(plot = False) 
                self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                self.value = torch.tensor([self.world.value])[:,None]
            
            
                self.state_sample = torch.zeros([1,100])
                self.action_sample= torch.zeros([1,2])
                self.h_post_lstm = torch.zeros([1,400])
                self.c_post_lstm = torch.zeros([1,400])
            
                
            
            
                #??? Here some additional sampling
                self.posterior()
                
                
                

                self.h_lstm = torch.zeros([1,400])
                self.c_lstm = torch.zeros([1,400])
            
            
                self.actionet()
                 
            
                for k in range(N_step):
                    
                    
                    self.critic()
                    
                    predicted_critic = self.predict_G.clone()
            
                
                    self.transition(N_samples = N_samples)
                    self.state_sample = self.state_trans_sample.detach().clone()
                    
                    
                    remember_one_state = self.state_sample[[0],:].clone()
                    #remember the state to reassign and take action
                    
                    next_step_entropy =(0.5+0.5*(np.log(2*np.pi)+self.state_logvar)).detach().clone()
                    
                    
                    
                    #self.likelihood() # MULTIPLE!!!!!!!! AND THEN BACK AGAIN!!!!
                    self.valuenet() 
                    #self.obs = self.obs_pb.clone()
                    
                    
                    #self.h_post_lstm = torch.tile(self.h_lstm,[N_samples,1])
                    #self.c_post_lstm = torch.tile(self.c_lstm,[N_samples,1])
                    #self.action_sample = torch.tile(self.action_sample,(N_samples,1))
                    #self.posterior()
                    #self.action_sample = self.action_sample[[0],:]
                    #self.h_lstm = self.h_lstm[[0],:]
                    #self.c_lstm = self.c_lstm[[0],:]
                    
                    r = torch.mean(self.predict_value.detach().clone(),dim = 0)  #NOT AS IN THEORY!!!
                    
                    #if torch.sum(self.predict_value>0.1) > 1:
                    if r >0.2:
                        a = 0
                        
                        #print(r)
                    #   print(r)#print(torch.max(self.predict_value))
                    #print(torch.max(self.predict_value))
                    
                    uncertainty_entropy = 0#(0.5+0.5*(np.log(2*np.pi)+torch.mean(self.state_logvar,dim = 0))).detach().clone()
                
                    self.actionet()
                    self.critic()
                    
                    decay_coef = 0.8
                    boots_G = torch.mean(self.predict_G,dim = 0)

                    #-0.0*(next_step_entropy-uncertainty_entropy)
                    G_est = -r+boots_G*decay_coef
                
                    loss = self.GLike(G_est.detach().clone(),predicted_critic)*100
                    
                    loss /= (N_trajectories*(N_step-1))
                    loss.backward()
                    
                    
                    self.state_sample = remember_one_state.detach().clone()
                  
                    self.actionet()
                    #self.action_entropy()
                
                    #self.critic()
                    
                    
                    cumulative_loss += loss.detach().numpy()
                
    
                    
          
                    
            self.critic_optimizer.step()
            loss_history[i] = cumulative_loss
            print(i, loss_history[i])
            
        plt.figure()
        plt.plot(loss_history)
        
        
    def Gplt(self):
        
        size = 20
        Ghist = np.zeros([size,size])
        
        dist_space = np.linspace(0.025,0.975,20)
        orient_space = np.linspace(-0.95,0.95,20)
        
        for i,dist_coord in enumerate(dist_space):
            for j,orient_coord in enumerate(orient_space):
                self.action_sample = torch.Tensor([[dist_coord],[orient_coord]]).T 
                self.critic()
                Ghist[j,i] = self.predict_G.detach().numpy()
                
        plt.figure()
        plt.imshow(Ghist,origin = 'lower')
     
        colorbar = plt.colorbar()
        colorbar.set_label('EFE')
        plt.xticks(np.arange(size)[::2],np.round(dist_space[::2],decimals = 1))
        plt.yticks(np.arange(size)[::2],np.round(orient_space[::2],decimals = 1))
        
        plt.xlabel('move forward')
        plt.ylabel('turn anticlockwise')
        
        
        
        
    def plot_EFE(self, N_trajectories = 1):
        bias = torch.from_numpy((np.array([0x6a,0xb8,0x28])/255.0).astype('float32'))[None,:,None,None]#np.random.rand(self.N_vis,self.N_vis)/10+0.7
        
      
        N_steps = 10
   
        counter = 0
      
   
        self.world.generate_world()        
        for i in range(N_trajectories):
            
    
            
            self.world.init_equal_angles_track()
            
           
                
            self.world.render_image(plot = True) 
            self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
            
            self.h_post_lstm = torch.zeros([1,400])
            self.c_post_lstm = torch.zeros([1,400])
            self.state_sample = torch.zeros([1,100])
            self.action_sample= torch.zeros([1,2])
            
            self.posterior()
            self.likelihood()
            
            
            
            self.h_lstm = torch.zeros([1,400])
            self.c_lstm = torch.zeros([1,400])
            
            
            
            self.world.vis_map()
            
            
            
            
            self.Gplt()
            
            self.actionet()
            self.world.action_step(self.action_sample[0,0].detach().numpy(),self.action_sample[0,1].detach().numpy())
            
            
            for j in range(N_steps):
                 
                self.world.render_image(plot = True)   
                self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                

                self.posterior()
                self.likelihood()
                self.valuenet()
                predicted_torch_image = self.obs_pb
                
                #plot imagery
                #self.plot_torch_obs(predicted_torch_image+bias) 
                

                self.world.vis_map()
                
                ##########
                
                self.actionet(10000)
                
                fig,ax = plt.subplots()
                H = ax.hist2d(self.action_sample[:,0].detach().numpy(), 
                                         self.action_sample[:,1].detach().numpy(), bins=20, 
                                         range = [[0, 1], [-1, 1]])
                colorbar = fig.colorbar(H[3])
                colorbar.set_label('total count')
                ax.set_aspect(0.5)
                
                
                plt.xlabel('move forward')
                plt.ylabel('turn anticlockwise')
                ###########
                
                self.Gplt()
                
                self.actionet()
                self.world.action_step(self.action_sample[0,0].detach().numpy(),self.action_sample[0,1].detach().numpy())
       
            
            print(i)
            
            #jump
    
    
    
    def retrain_value(self):
        
    
        # try other learning rates?????????
        # for the first traing - 0.0003, for the fine aftertraining = 0.00003
        self.optimizer = torch.optim.Adam([
    {'params': self.value_fc0.parameters()},
    {'params': self.value_fc1_.parameters()},
    {'params': self.value_fc2_.parameters()}], lr=0.0003)
       
        bias = torch.from_numpy((np.array([0x6a,0xb8,0x28])/255.0).astype('float32'))[None,:,None,None]#np.random.rand(self.N_vis,self.N_vis)/10+0.7
        
        N_rollouts = 200#00_000
        N_starts_per_world = 6
        N_steps = 10
        #BatchSize = 20
        
        counter = 0
       
        loss_history_VALUE = np.zeros(N_rollouts)
        
        
        
        
        for i in range(N_rollouts):
            
            self.optimizer.zero_grad()
            self.world.generate_world()
            
            
            
            loss = torch.tensor(0.0)
            
            for k in range(N_starts_per_world):
                self.world.init_random_pos()
                
                
                self.world.render_image(plot = False) 
                self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                self.value = torch.tensor([self.world.value])[:,None]
                self.state_sample = torch.zeros([1,100])
                self.action_sample= torch.zeros([1,2])
                self.h_post_lstm = torch.zeros([1,400])
                self.c_post_lstm = torch.zeros([1,400])
                
                
                self.posterior()
              
                self.valuenet()
                
                
                
                self.actionet()
                self.world.action_step(self.action_sample.detach().numpy()[0,0],self.action_sample.detach().numpy()[0][1])
                
                loss += self.GLike(self.value, self.predict_value)*1.0
                loss_VALUE = 0
                
                
                self.h_lstm = torch.zeros([1,400])
                self.c_lstm = torch.zeros([1,400])
                
                for j in range(N_steps):
                    
                              
                    
                    
                   
                    self.world.render_image(plot = False)
                        
                    self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                    self.value = torch.tensor([self.world.value])[:,None]
                    
                    
                    #transition goes first !!!!!!!!!!!!! This is prediction from previous step 
                    
                    self.transition()
                    #self.state_trans_sample = torch.randn_like(self.state_sample)
                    
                    self.posterior()
                    #self.state_sample = self.state_sample.detach().clone() #?$
                    
                    self.valuenet()
                    
                    if self.value.detach().numpy()[0][0] > 0.0:
                        print('est: ',self.predict_value)
                    
                    
                    self.actionet()
                    self.action_sample = self.action_sample.detach().clone()#?$
               
                    self.world.action_step(self.action_sample.detach().numpy()[0,0],self.action_sample.detach().numpy()[0,1])
                    
                    
                    loss += self.GLike(self.value, self.predict_value)*1.0
                    
                    
                    
                    loss_VALUE += self.GLike(self.value, self.predict_value)
                    
                   
                 
        
            loss = loss/(N_starts_per_world*N_steps)
            loss.backward()  
            self.optimizer.step()  
            
            loss_VALUE /= (N_starts_per_world*N_steps)
            
            
            loss_history_VALUE[counter] = loss_VALUE.detach().numpy()
            print(loss_history_VALUE[counter])
            counter += 1
            print(i)
                
                    
            #if i%BatchSize==BatchSize-1:
            #    print(i)    
            #    print(loss_history[counter-1])
                    
                #if (i == 4000):
                    
                #    for param_group in self.optimizer.param_groups:
                #        param_group['lr'] = 0.00001
                    
                        

            
        
        plt.figure()
        plt.plot(loss_history_VALUE)
        plt.title('VALUE loss history')
                
                
              
        
    def plot_trajectories(self, N_trajectories = 10,N_visible = 10, random_action = False):
        # TODO: add reward calculation, flowers coord accumulation
        # calculate reward for all r
        
        bias = torch.from_numpy((np.array([0x6a,0xb8,0x28])/255.0).astype('float32'))[None,:,None,None]#np.random.rand(self.N_vis,self.N_vis)/10+0.7
        
      
        N_steps =5
   
        counter = 0
      
        
        trajectory_store = np.zeros((N_steps+1, N_visible,2))
        cum_reward = 0
        violet_flower_store = np.zeros((N_visible,2))
        red_flower_store = np.zeros((N_visible,2))
        colors = np.ones((N_visible,3))*0.9
    
        
        
        # make a mean value for the flowers
        for i in range(N_trajectories):
            self.world.generate_world() 
            
            
            #self.world.init_equal_angles_track()
            self.world.init_random_pos()
            
            
            if i<N_visible:
                trajectory_store[0,i,:] = self.world.cur_pos[:]
                violet_flower_store[i,:] = self.world.honey_flowers_coord[0]
                red_flower_store[i,:] = self.world.honey_flowers_coord[1]
            

                
            self.world.render_image(plot = False) 
            self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
            
            self.h_post_lstm = torch.zeros([1,400])
            self.c_post_lstm = torch.zeros([1,400])
            self.state_sample = torch.zeros([1,100])
            self.action_sample= torch.zeros([1,2])
            
            self.posterior()
            self.likelihood()
            
            
            
            self.h_lstm = torch.zeros([1,400])
            self.c_lstm = torch.zeros([1,400])
            
            if not random_action:
                self.actionet()
            else:
                dist_coord = np.random.rand()
                orient_coord = np.random.rand()*2-1.0   
                self.action_sample = torch.Tensor([[dist_coord],[orient_coord]]).T  
            self.world.action_step(self.action_sample[0,0].detach().numpy(),self.action_sample[0,1].detach().numpy())
            
            
            for j in range(N_steps):
                
                if i<N_visible:
                    trajectory_store[j+1,i,:] = self.world.cur_pos[:]
                     
                self.world.render_image(plot = False)   
                self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                self.value = torch.tensor([self.world.value])[:,None]
                cum_reward += self.value.detach().numpy()[0,0]
                
                if i<N_visible:
                    if self.value > 0:
                        colors[i,:] += -0.45

                self.posterior()
                #self.likelihood()
                self.valuenet()
                
                #predicted_torch_image = self.obs_pb
                
                #plot imagery
                #self.plot_torch_obs(predicted_torch_image+bias) 
                
                
                if not random_action:
                    self.actionet()
                else:
                    dist_coord = np.random.rand()
                    orient_coord = np.random.rand()*2-1.0   
                    self.action_sample = torch.Tensor([[dist_coord],[orient_coord]]).T  
                self.world.action_step(self.action_sample[0,0].detach().numpy(),self.action_sample[0,1].detach().numpy())
       
            if i%100 == 0:
                print(i)
            counter += 1
            
            #jump
        fig,ax = plt.subplots()
        ax.scatter(violet_flower_store[:,0],violet_flower_store[:,1],s = 20, c = 'violet')
        ax.scatter(red_flower_store[:,0],red_flower_store[:,1],s = 20, c = 'red')
        
       
        for i in range(N_visible):
            plt.plot(trajectory_store[:,i,0],trajectory_store[:,i,1],color = colors[i,:])
            
        
        plt.xlim(-150,150)
        plt.ylim(-150,150)
        ax.set_aspect('equal')
        
        
        cum_reward  /= N_trajectories
        print(cum_reward)
        return cum_reward 
        
            
                    
   

        
        
    def test_FE(self, moves = None):
     
        bias = torch.from_numpy((np.array([0x6a,0xb8,0x28])/255.0).astype('float32'))[None,:,None,None]#np.random.rand(self.N_vis,self.N_vis)/10+0.7
        
        N_rollouts = 1
        N_steps = 10
        if moves is not None:
            N_steps = len(moves)
   
        counter = 0
      
        
        
        for i in range(N_rollouts):
            
    
            self.world.generate_world()
            self.world.init_the_same_track()
            
            
            
         
                
            self.world.render_image(plot = True) 
            plt.xticks([])
            plt.yticks([])
            self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
            
            self.h_post_lstm = torch.zeros([1,400])
            self.c_post_lstm = torch.zeros([1,400])
            self.state_sample = torch.zeros([1,100])
            self.action_sample= torch.zeros([1,2])
            
            self.posterior()
            self.likelihood()
            self.valuenet()
            
            
            
            self.h_lstm = torch.zeros([1,400])
            self.c_lstm = torch.zeros([1,400])
            
            
            
            for j in range(N_steps):
                     
                self.world.render_image(plot = False)   
                self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                dist_coord = np.random.rand()
                orient_coord = np.random.rand()*2-1.0   
                print(dist_coord,orient_coord)
                
                
                self.action_sample = torch.Tensor([[dist_coord],[orient_coord]]).T  
                if moves is not None:
                    self.action_sample = torch.Tensor([moves[j]])
                
                #only for the compas value
                # (suppose we have ideal transition model for the compas value depending only on the
                #init value and taken action)
                self.world.action_step(dist_coord,orient_coord)
                    #transition goes first !!!!!!!!!!!!!
                self.transition()
                self.state_sample = self.state_trans_sample
                self.likelihood()
                self.valuenet()
                

                predicted_torch_image = self.obs_pb
                
                #plot imagery
                self.plot_torch_obs(predicted_torch_image+bias) 
                plt.xticks([])
                plt.yticks([])
                #
                    
            print(i)
                
        
    
        

    def forced_train_FE_separate(self):
        
    
        # try other learning rates?????????
        # for the first traing - 0.0003, for the fine aftertraining = 0.00003
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
       
        bias = torch.from_numpy((np.array([0x6a,0xb8,0x28])/255.0).astype('float32'))[None,:,None,None]#np.random.rand(self.N_vis,self.N_vis)/10+0.7
        
        N_rollouts = 1_000_000#00_000
        N_starts_per_world = 6
        N_steps = 10
        #BatchSize = 20
        
        counter = 0
        loss_history = np.zeros(N_rollouts)
        loss_history_GKL = np.zeros(N_rollouts)
        loss_history_MSE = np.zeros(N_rollouts)
        loss_history_VALUE = np.zeros(N_rollouts)
        
        
        
        freq_plot =99#199#49
        for i in range(N_rollouts):
            
            self.optimizer.zero_grad()
            self.world.generate_world()
            
            
            
          
            
            loss = torch.tensor(0.0)
            loss_MSE = torch.tensor(0.0)
            loss_GKL = torch.tensor(0.0)
            loss_VALUE = torch.tensor(0.0)
            
            for k in range(N_starts_per_world):
                self.world.init_random_pos()
                
                
                self.world.render_image(plot = False) 
                self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                self.value = torch.tensor([self.world.value])[:,None]
                
                self.state_sample = torch.zeros([1,100])
                self.action_sample= torch.zeros([1,2])
                self.h_post_lstm = torch.zeros([1,400])
                self.c_post_lstm = torch.zeros([1,400])
                
                
                self.posterior()
                self.likelihood()
                self.valuenet()
                
                
                
                dist_coord = np.random.rand()
                orient_coord = np.random.rand()*2-1.0   
                self.action_sample = torch.Tensor([[dist_coord],[orient_coord]]).T  
                self.world.action_step(dist_coord,orient_coord)
                
                loss += self.GLike(self.obs,self.obs_pb)#self.BLike(self.obs,self.obs_pb)
                #loss += 0.02*self.GDKLoss(self.state_mean,self.state_logvar,self.state_mean.detach().clone(),torch.ones_like(self.state_mean.detach().clone())*(-10.0))
                loss += self.GLike(self.value, self.predict_value)*1.0
                
                
                loss_MSE += self.GLike(self.obs,self.obs_pb)
                loss_VALUE += self.GLike(self.value, self.predict_value)
                
      
                self.h_lstm = torch.zeros([1,400])
                self.c_lstm = torch.zeros([1,400])
                
                for j in range(N_steps):
                    
                              
                    
                    
                    if i %freq_plot == 0:    
                        self.world.render_image(plot = True)
                        self.world.vis_map()
                    else:
                        self.world.render_image(plot = False)
                        
                    self.obs = torch.from_numpy(self.world.image.astype('float32').transpose((2,0,1)))[None,:,:,:]-bias
                    self.value = torch.tensor([self.world.value])[:,None]
                    
                    
                    #transition goes first !!!!!!!!!!!!! This is prediction from previous step 
                    self.transition()
                    #self.state_trans_sample = torch.randn_like(self.state_sample)
                    
                    

                    self.posterior()
                    self.likelihood()
                    self.valuenet()
                    
                    if self.value.detach().numpy()[0][0] > 0.0:
                        print('est: ',self.predict_value)
                    
                    
                    dist_coord = np.random.rand()
                    orient_coord = np.random.rand()*2-1.0   
                    self.action_sample = torch.Tensor([[dist_coord],[orient_coord]]).T   
                    self.world.action_step(dist_coord,orient_coord)
                    
                    
                    loss += self.GLike(self.obs,self.obs_pb)#self.BLike(self.obs,self.obs_pb)
                    loss += self.GLike(self.value, self.predict_value)*1.0
                    #loss += 0.02*self.GDKLoss(self.state_mean,self.state_logvar,self.state_mean,torch.ones_like(self.state_mean)*(-10.0))
                    loss += 0.05*self.GDKLoss(self.state_mean,self.state_logvar,self.state_trans_mean,self.state_trans_logvar)#(self.state_pb, self.state_trans_pb)
                    #loss += ????
                    #if i <500:
                    #    pass
                        #loss += 0.01*self.GDKLoss(self.state_mean,self.state_logvar,torch.zeros_like(self.state_mean),torch.zeros_like(self.state_logvar))#(self.state_pb, self.state_trans_pb)
                        #loss += 0.01*self.GDKLoss(self.state_mean,self.state_logvar,self.state_trans_mean,self.state_trans_logvar)#(self.state_pb, self.state_trans_pb)
                    #loss += 0.001*self.GDKLoss(self.state_mean,self.state_logvar,torch.zeros_like(self.state_mean),torch.zeros_like(self.state_logvar))#(self.state_pb, self.state_trans_pb)
                    
                    #else:
                    #    loss += 0.01*self.GDKLoss(self.state_mean,self.state_logvar,self.state_trans_mean,self.state_trans_logvar)#(self.state_pb, self.state_trans_pb)
                    #loss += 0.001*self.GDKLoss(self.state_mean,self.state_logvar,self.state_trans_mean,self.state_trans_logvar)#(self.state_pb, self.state_trans_pb)
                 
                 
                    
                    
                    
                    loss_MSE += self.GLike(self.obs,self.obs_pb)
                    loss_GKL += self.GDKLoss(self.state_mean,self.state_logvar,self.state_trans_mean,self.state_trans_logvar)#(self.state_pb, self.state_trans_pb)
                    loss_VALUE += self.GLike(self.value, self.predict_value)
                    
                    #if i < 1000000:
                    #    loss += 0.01*self.GDKLoss(self.state_mean,self.state_logvar,torch.tensor(0),torch.tensor(0))#(self.state_pb, self.state_trans_pb)
            
                   
                    
                    predicted_torch_image = self.obs_pb
                    
                    if i%freq_plot == 0: 
                        self.plot_torch_obs(predicted_torch_image+bias)
                    
                    # ONLY FOR TEST
                    #self.state_sample = self.state_trans_sample#.detach().clone()
                    #self.transition()
                    #self.state_sample = self.state_trans_sample
                    #self.likelihood()
                    
                    #predicted_torch_image = self.obs_pb
                    
                    #if i%freq_plot == 0: 
                    #    self.plot_torch_obs(predicted_torch_image+bias)
                    
                    # ONLY FOR TEST
                    
                   
                    
                 
        
            loss = loss/(N_starts_per_world*N_steps)
            loss.backward()  
            self.optimizer.step() 
            
            
            loss_MSE /= (N_starts_per_world*N_steps)
            loss_GKL /= (N_starts_per_world*N_steps)
            loss_VALUE /= (N_starts_per_world*N_steps)
            
            
            loss_history_MSE[counter] = loss_MSE.detach().numpy()
            loss_history_GKL[counter] = loss_GKL.detach().numpy()
            loss_history_VALUE[counter] = loss_VALUE.detach().numpy()
            print(loss_history_MSE[counter],loss_history_GKL[counter],loss_history_VALUE[counter])
            counter += 1
            print(i)
                
                    
            #if i%BatchSize==BatchSize-1:
            #    print(i)    
            #    print(loss_history[counter-1])
                    
                #if (i == 4000):
                    
                #    for param_group in self.optimizer.param_groups:
                #        param_group['lr'] = 0.00001
                    
                        

            
        plt.figure()
        plt.plot(loss_history_MSE)
        plt.title('MSE loss history')  
        plt.figure()
        plt.plot(loss_history_GKL)
        plt.title('GKL loss history')
        plt.figure()
        plt.plot(loss_history_VALUE)
        plt.title('GKL loss history')
                
                
                
            
        
        
    #def forced_test_FE(self):
    #    pass
    
    '''
    def posterior(self, sampling = True):
        
        
        x = self.obs.clone()
        

        x = self.post_conv0(x)
        x = torch.nn.functional.relu(x)
        x = self.post_conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.post_conv2(x)
        x = torch.nn.functional.relu(x)
        x = x[:,:,0,0]
        x = self.post_fc3(x)
        x = torch.nn.functional.relu(x)
        self.state_mean = self.post_fc4_mu(x)
        self.state_logvar = self.post_fc4_var(x)

        epsilon = torch.randn_like(self.state_mean)

        self.state_sample = self.state_mean + torch.exp(0.5 * self.state_logvar) * epsilon
       
        
        
        
        
        #x = torch.concatenate([self.state_sample.clone(),self.decision_sample.clone(), x], dim = 1)
    '''
      
    
    def critic(self):
        x = torch.concatenate([self.state_sample.clone(),self.action_sample.clone()],dim = 1)
        
        
        
        x = torch.nn.functional.leaky_relu(self.critic_fc0(x))
        x1 = torch.nn.functional.sigmoid(self.critic_fc1_sig(x))
        x2 = torch.nn.functional.leaky_relu(self.critic_fc1_rel(x))
        x = self.critic_fc2_(x1*x2)
        x = -torch.mean(torch.nn.functional.leaky_relu(x),dim = 1, keepdim=True)
        #x = -(1/4)*torch.nn.functional.softplus(4*(x-1))
        
        self.predict_G = x

        
    def valuenet(self, sampling = True):
        x = self.state_sample.clone()
        
        x = torch.nn.functional.leaky_relu(self.value_fc0(x))
        x = torch.nn.functional.leaky_relu(self.value_fc1_(x))
        x = torch.nn.functional.leaky_relu(self.value_fc2_(x))
        
        #x = torch.sigmoid(x)*1.1-0.05#-0.1
        #x = torch.nn.functional.relu(x)
        
        #x = self.value_fc1(x)
        #x = torch.nn.functional.relu(x)
        
        
        #x = self.value_fc2(x)
   
      
        self.predict_value = x#torch.sigmoid(x)
 
        
        
    
    def posterior(self, sampling = True, N_samples = 1):
        
        
        x = self.obs.clone()
        

        x = self.post_conv0(x)
        #self.post_batch0.apply(x)
        #x = (x-self.post_batch0.mean)/torch.sqrt(self.post_batch0.var)
        x = torch.nn.functional.leaky_relu(x)
        
        x = self.post_conv1(x)
        #self.post_batch0.apply(x)
        #x = (x-self.post_batch0.mean)/torch.sqrt(self.post_batch0.var)
        x = torch.nn.functional.leaky_relu(x)
        
        
        x,self.idx1 = self.post_max1(x)
        
        x = self.post_conv2(x)
        #self.post_batch0.apply(x)
        #x = (x-self.post_batch0.mean)/torch.sqrt(self.post_batch0.var)
        x = torch.nn.functional.leaky_relu(x)
        x,self.idx2 = self.post_max2(x)
        
        
        

        x = torch.flatten(x,start_dim = 1)
        
        
    
        x_lstm = torch.concatenate([self.state_sample.clone(),self.action_sample.clone()],dim = 1)
        x_lstm =self.post_lstm_fc0(x_lstm)
        x_lstm  = torch.nn.functional.leaky_relu(x_lstm)
        
        self.h_post_lstm, self.c_post_lstm = self.post_lstm(x_lstm, (self.h_post_lstm, self.c_post_lstm))
        
        x_lstm = self.h_post_lstm
        
        self.h_post_lstm= torch.zeros_like(self.h_post_lstm)
        self.c_post_lstm= torch.zeros_like(self.c_post_lstm)
        
        
        
        x = torch.concatenate([x,x_lstm],dim = 1)
        
        
        
        x = self.post_fc5(x)
        x = torch.nn.functional.leaky_relu(x)
  
     
        self.state_mean = self.post_fc6_mu(x)
        self.state_logvar = self.post_fc6_var(x)

        #self.state_sample = self.state_mean 
        epsilon = torch.randn((N_samples,self.state_mean.size(1)))
        self.state_sample = self.state_mean + torch.exp(0.5 * self.state_logvar) * epsilon
        
        
        #x = torch.concatenate([self.state_sample.clone(),self.decision_sample.clone(), x], dim = 1)
        
    def likelihood(self, sampling = True):
        
        
        x = self.state_sample.clone()
        
        x = self.like_fc0(x)
        #self.like_batch0.apply(x)
        #x = (x-self.like_batch0.mean)/torch.sqrt(self.like_batch0.var)
        x = torch.nn.functional.leaky_relu(x)
        
        x = self.like_fc1(x)
        #self.like_batch0.apply(x)
        #x = (x-self.like_batch0.mean)/torch.sqrt(self.like_batch0.var)
        x = torch.nn.functional.leaky_relu(x)
        
        x = torch.reshape(x,(-1,160,6,6))
        
        x = self.like_upsample1(x)
        
        x = self.like_decon2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.like_upsample2(x)
        
     
        
        x = self.like_decon5(x)
        x = torch.nn.functional.leaky_relu(x)
        
        x = self.like_decon6(x)
        #self.like_batch1.apply(x)
        #x = (x-self.like_batch1.mean)/torch.sqrt(self.like_batch1.var)
       

        self.obs_pb = x#torch.sigmoid(x)
        
    '''
    def likelihood(self, sampling = True):
        
        
        x = self.state_sample.clone()
        
        x = self.like_fc0(x)
        x = torch.nn.functional.relu(x)
        x = self.like_fc1(x)
        x = torch.nn.functional.relu(x)
        
        x = x[:,:,None,None]

        x = self.like_decon2(x)
        x = torch.nn.functional.relu(x)
        
        x = self.like_decon3(x)
        x = torch.nn.functional.relu(x)
        
        x = self.like_decon4(x)
        self.obs_pb = torch.sigmoid(x)
    '''
    
    def action_entropy(self):
        x = self.action_sample.clone()+torch.randn_like(self.action_sample.clone().detach())*torch.Tensor([[1.0,2.0]])*0.1
        x = torch.concatenate([self.state_sample.clone(),x],dim = 1)
        
        x = self.recon_fc0(x)
        x = torch.nn.functional.relu(x)
        x = self.recon_fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.recon_fc2(x)
        
        self.intention_recon = x
    
    def actionet(self, N_samples = 1):
        
        if N_samples > self.state_sample.size(0):
            self.intention_noise = torch.randn([N_samples,2])
        else:
            self.intention_noise = torch.randn([self.state_sample.size(0),2])
        #if self.intention_noise.size(0) == self.state_sample.size(0):
            
        
        x = torch.concatenate([torch.tile(self.state_sample.clone(),(N_samples,1)),self.intention_noise.clone()],dim = 1)
        

        x = self.habit_fc0_(x)
        x = torch.nn.functional.relu(x)
        x = self.habit_fc1_(x)
        x = torch.nn.functional.relu(x)
        x = self.habit_fc2_(x)
        
        x1 = torch.nn.functional.sigmoid(x[:,[0]])#torch.clip(x[:,[0]],0,1)
        x2 = torch.nn.functional.tanh(x[:,[1]])#torch.clip(x[:,[1]],-1,1)
        
        
        self.action_sample = torch.concatenate([x1,x2],dim = 1)
  
    
    def transition(self,sampling = True, N_samples = 1):
        x = torch.concatenate([self.state_sample.clone(),self.action_sample.clone()],dim = 1)
        x =self.trans_fc0(x)
        x  = torch.nn.functional.leaky_relu(x)
        
        self.h_lstm, self.c_lstm = self.trans_lstm(x, (self.h_lstm, self.c_lstm))
        x = self.h_lstm
        #x = self.trans_fc0(x)
        #x = torch.nn.functional.relu(x)
        #x = self.trans_fc1(x)
        #x = torch.nn.functional.relu(x)
        
        
        self.state_trans_mean = self.trans_fc1_mu(x)
        self.state_trans_logvar = self.trans_fc1_var(x)
        
        epsilon = torch.randn((N_samples,self.state_mean.size(1)))
        ## JUMP
        self.state_trans_sample = self.state_trans_mean + torch.exp(0.5 * self.state_trans_logvar) * epsilon
  
    
        
    '''
    def habit(self, sampling = True):
        x = self.state_sample.clone() 
        x = self.habit_fc0(x)
        x = torch.nn.functional.relu(x)
        self.intention_sample = torch.sigmoid(self.habit_fc1(x))
        
        
        
       
    def transition(self,sampling = True):
        x = torch.concatenate([self.state_sample.clone(),self.intention_sample.clone()],dim = 1)
        x = self.trans_fc0(x)
        x = torch.nn.functional.relu(x)
        self.state_trans_mean = self.trans_fc1_mu(x)
        self.state_trans_logvar = self.trans_fc1_var(x)
        
        epsilon = torch.randn_like(self.state_mean)
        self.state_trans_sample = self.state_trans_mean + torch.exp(0.5 * self.state_trans_logvar) * epsilon
    ''' 

        
        
        
        
        #x = torch.concatenate([self.state_sample.clone(),self.decision_sample.clone(), x], dim = 1)
        
             
        
        
   
       
  


class World:
    
    def __init__(self):
        
        self.N_vis = 100  #size of visual scene
        self.R = self.N_vis/2
        self.halfW = self.N_vis/2-0.5
        self.violet_r = 2.0
        self.red_r = 1.0
        
        
        
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
            
        
        
    def vis_map(self):
        fig,ax = plt.subplots()
        ax.scatter(self.honey_flowers_coord[0,0],self.honey_flowers_coord[0,1],s = 20, c = 'violet')
        ax.scatter(self.honey_flowers_coord[1,0],self.honey_flowers_coord[1,1],s = 20, c = 'orange')
        ax.scatter(self.cur_pos[0],self.cur_pos[1],s = 25, c = 'red')
        ax.scatter(self.cur_pos[0]+np.cos(self.cur_orient)*50,self.cur_pos[1]+np.sin(self.cur_orient)*40,s = 25, c = 'green')
        plt.legend([str(self.value)])
        plt.xlim(-150,150)
        plt.ylim(-150,150)
        ax.set_aspect('equal')
        
        
    
        
        
    def generate_world(self, max_N_taps = 500):
        
        
        self.background_color = np.array([0x6a,0xb8,0x28])
        
        cur_orient = np.random.rand()*2*np.pi
        self.start_track_orient = cur_orient
        start_coord = np.zeros((2,))
        self.start_coord = start_coord.copy()
        
    
        self.flower_circle_rad = 100#200
        self.border_circle_rad=  120#500
        
        
        # dissipation rate of value (~opacity changing rate, when bee is located at maximum)
        self.value_dispeed = 0.50001
        
       
        self.honey_flowers_coord = list()
        self.flowers_type = list()
        
        
    
        self.flow_center_interp_points = list()
        
        
        #list of array of arrays
        self.flow_petals_interp_points = list()
        
        
        
        ori = np.pi+(np.random.rand()-0.5)*(np.pi*2/3)
        
        self.rem_flow_ori0 = ori
        self.honey_flowers_coord.append(np.array((np.cos(ori),np.sin(ori)))*self.flower_circle_rad)
        self.flowers_type.append(0)
        
        ori = (np.random.rand()-0.5)*(np.pi*2/3)
        
        self.rem_flow_ori1 = ori
        self.honey_flowers_coord.append(np.array((np.cos(ori),np.sin(ori)))*self.flower_circle_rad)
        self.flowers_type.append(1)
                #if np.random.rand() < 0.5:
                #    self.flowers_type.append(0)
                #else:
                #    self.flowers_type.append(1)
        for j in range(len(self.flowers_type)) :
            axes = (7+np.random.randint(4)-1, 7+np.random.randint(4)-1)
            angle = 0
            startAngle = int(np.random.rand()*360)
            endAngle = int(startAngle+360)
    
            center = self.honey_flowers_coord[j].astype('int')
    
                    # Generate ellipse points
            points = cv2.ellipse2Poly(center, axes, angle, startAngle, endAngle, delta=90).astype('double')
    
            jitter = np.random.rand(points.shape[0],points.shape[1])*2-1
            points += jitter
                    
                    
            num_points = 10
            tck, u = scipy.interpolate.splprep(points.T, k=2, s=0, per=True)
            u_new = np.linspace(u.min(), u.max(), num_points)
            x_new, y_new = scipy.interpolate.splev(u_new, tck)
            self.flow_center_interp_points.append((np.vstack((x_new, y_new)).T))
                    
                    
            axes = (20+np.random.randint(3)-1,20+np.random.randint(3)-1)
            angle = int(np.random.rand()*360)
            startAngle = int(np.random.rand()*360)
            endAngle =startAngle +360
            numPetals = np.random.randint(5,9)
            delta = int(360/numPetals)
            self.petals_centers = cv2.ellipse2Poly(center, axes, angle, startAngle, endAngle, delta= delta).astype('double')
                    
            self.flow_petals_interp_points.append(list())
                    
            for k in range(numPetals):
                        
                axes = (7,14)
                vec = self.petals_centers[k]- center
                         
                angle = int(np.arctan2(vec[1],vec[0])*180/np.pi+90)
                startAngle = int(np.random.rand()*360)
                endAngle =startAngle +360
                        
                pre_interp = cv2.ellipse2Poly(self.petals_centers[k].astype('int'), axes, angle, startAngle, endAngle, delta= 60).astype('double')
                        
                pre_interp += (np.random.randint(0,5,size = pre_interp.shape)-2)
                        
                        
                num_points = 12
                tck, u = scipy.interpolate.splprep(pre_interp.T, k=2, s=0, per=True)
                u_new = np.linspace(u.min(), u.max(), num_points)
                x_new, y_new = scipy.interpolate.splev(u_new, tck)
                self.flow_petals_interp_points[-1].append((np.vstack((x_new, y_new)).T))
                        
         
      
         
            
        self.honey_flowers_coord = np.array(self.honey_flowers_coord)
        
        self.violet_max_value = 1.0
        
        # opacity of flowers with "nectar"
        self.violet_flower_charge = np.ones(len(self.honey_flowers_coord))*1.0
        self.was_visited = np.array([False,False],dtype = 'bool')
        
        
        
        
        
        
        
        
        #### IF YOU WANT TO VIZ PATH!!!!
        
        
        #plt.figure()
        #plt.plot(self.coord_list[:,0],self.coord_list[:,1])
        
        
        
        #self.max_speed_linear = 10.0
        #self.max_speed_angle = 0.3
        
  
        self.cur_pos = self.start_coord
        self.cur_orient = self.start_track_orient#np.arctan2(self.coord_list[1,0]-self.coord_list[0,0],self.coord_list[1,1]-self.coord_list[0,1])  #in pi
        
        
        
    def add_flower(self, valid_idx):
        for idx in valid_idx:
            
            


            # Draw a smooth interpolation through the points
            
            # Create a black canvas and draw the polyline
            #canvas = np.zeros((512, 512, 3), dtype=np.uint8)
            #cv2.polylines(canvas, [interp_points], False, 255, thickness=2, lineType=cv2.LINE_AA)

            #pattern = np.zeros((10, 10), dtype=np.uint8)
            #for i in range(10):
            #    pattern[i, i] = 255

            # Create a mask with the pattern
            #mask = np.tile(pattern, (10, 10))

            #canvas = cv2.bitwise_and(canvas, canvas, mask=mask)

            #self.interp_points.append((np.vstack((x_new, y_new)).T-self.cur_pos[None,:]+self.R).astype(np.int32))
            
            #points of the flower center part
            flow_centered_x = self.flow_center_interp_points[idx][:,0]-self.cur_pos[0]
            flow_centered_y =  self.flow_center_interp_points[idx][:,1]-self.cur_pos[1]
                                                                
            rot_flow_x = np.cos(self.cur_orient)*flow_centered_x+np.sin(self.cur_orient)*flow_centered_y
            rot_flow_y = -np.sin(self.cur_orient)*flow_centered_x+np.cos(self.cur_orient)*flow_centered_y
            
            
            
            #rot_interp_points = 
            
            full_part = self.violet_flower_charge[idx]
            empty_part = (1.0-self.violet_flower_charge[idx])
            empty_image = self.image.copy()
            
            if self.flowers_type[idx] == 0:
                cur_color = np.array((0, 0, 139/255))#self.violet_flower_charge[idx]*np.array((0, 0, 139/255))+(1.0-self.violet_flower_charge[idx])*(self.background_color/255.0)
            else:
                cur_color = np.array((0x03/255, 0xD5/255, 0xFB/255))#self.violet_flower_charge[idx]*np.array((0, 0, 139/255))+(1.0-self.violet_flower_charge[idx])*(self.background_color/255.0)
            
            cv2.fillPoly(self.image, [np.array([rot_flow_x+self.R,rot_flow_y+self.R]).T.astype('int')], cur_color)
            if self.flowers_type[idx] == 0:
                cur_color = np.array((100/255, 0/255, 0/255))#self.violet_flower_charge[idx]*np.array((100/255, 0/255, 0/255))+(1.0-self.violet_flower_charge[idx])*(self.background_color/255.0)
            else:
                cur_color = np.array((0xC0/255, 0xC0/255, 0xC0/255))#self.violet_flower_charge[idx]*np.array((100/255, 0/255, 0/255))+(1.0-self.violet_flower_charge[idx])*(self.background_color/255.0)
            
                
            cv2.polylines(self.image, [np.array([rot_flow_x+self.R,rot_flow_y+self.R]).T.astype('int')], True, cur_color, 1) 

            for k in range(len(self.flow_petals_interp_points[idx])):
                flow_centered_x = self.flow_petals_interp_points[idx][k][:,0]-self.cur_pos[0]
                flow_centered_y =  self.flow_petals_interp_points[idx][k][:,1]-self.cur_pos[1]
                                                                    
                rot_flow_x = np.cos(self.cur_orient)*flow_centered_x+np.sin(self.cur_orient)*flow_centered_y
                rot_flow_y = -np.sin(self.cur_orient)*flow_centered_x+np.cos(self.cur_orient)*flow_centered_y
                if self.flowers_type[idx] == 0:
                    cur_color = np.array((0xe3/255,0x93/255,0xf8/255))#self.violet_flower_charge[idx]*np.array((0xe3/255,0x93/255,0xf8/255))+(1.0-self.violet_flower_charge[idx])*(self.background_color/255.0)
                else:
                    cur_color = np.array((255/255, 0/255, 0/255))#self.violet_flower_charge[idx]*np.array((0xe3/255,0x93/255,0xf8/255))+(1.0-self.violet_flower_charge[idx])*(self.background_color/255.0)
                
                cv2.fillPoly(self.image, [np.array([rot_flow_x+self.R,rot_flow_y+self.R]).T.astype('int')], cur_color)
                if self.flowers_type[idx] == 0:
                    cur_color = np.array((0xe3/500,0x93/500,0xf8/500))#self.violet_flower_charge[idx]*np.array((0xe3/500,0x93/500,0xf8/500))+(1.0-self.violet_flower_charge[idx])*(self.background_color/255.0)
                else:
                    cur_color = np.array((0x80/255, 0x0/255, 0x0/255))#self.violet_flower_charge[idx]*np.array((0xe3/500,0x93/500,0xf8/500))+(1.0-self.violet_flower_charge[idx])*(self.background_color/255.0)
                
                    
                cv2.polylines(self.image, [np.array([rot_flow_x+self.R,rot_flow_y+self.R]).T.astype('int')], True, cur_color, 1) 
                
            self.image = self.image*full_part+empty_image*empty_part
            

            
    def init_the_same_track(self):
        
        self.cur_orient = self.start_track_orient
        self.cur_pos = self.start_coord.copy()
        
        self.violet_flower_charge = np.ones(len(self.honey_flowers_coord))*1.0
        self.was_visited = np.array([False,False], dtype = 'bool')
        
        
    def init_equal_angles_track(self):
   
        self.cur_orient = (self.rem_flow_ori0 +self.rem_flow_ori1)/2.0
        self.cur_pos = self.start_coord.copy()
        
        self.violet_flower_charge = np.ones(len(self.honey_flowers_coord))*1.0
        self.was_visited = np.array([False,False], dtype = 'bool')
        
        
        
        
    def init_random_pos(self):
        
    
        self.cur_orient =np.random.rand()*2*np.pi
        self.cur_pos = self.start_coord.copy()
       
        self.violet_flower_charge = np.ones(len(self.honey_flowers_coord))*1.0
        self.was_visited = np.array([False,False], dtype = 'bool')
        
        
        
    def action_step(self, dist_coord, orient):
        
        
        
        
        #speed_linear = speed_linear_01*self.max_speed_linear
        #speed_angle =(speed_angle_01-0.5)*2*self.max_speed_angle
        
        
        # smoothing state
        
        noise_level_orient = np.clip(np.random.randn()*0.008,-0.01,0.01)  # multiplicative noise
        noise_level_dist = np.clip(np.random.randn()*0.02,-0.03,0.03)
        
        
        orient *= (1.0+noise_level_orient)
        dist_coord *= (1.0+noise_level_dist)
      
            
        if dist_coord > 1.0:
            dist_coord = 1.0
            
        if dist_coord < 0:
            dist_coord = 0
            
        if orient > 1.0:
            orient = 1.0
            
        if orient < -1.0:
            orient = -1.0
            
            
   
        self.cur_orient += orient*np.pi/2
               
      
            
        self.cur_pos += self.halfW*np.array([dist_coord*np.cos(self.cur_orient),dist_coord*np.sin(self.cur_orient)])
        
  
            
        if (self.active_flower is not None):
            self.violet_flower_charge[self.active_flower] -= self.value_dispeed
            if self.violet_flower_charge[self.active_flower]<0:
                self.violet_flower_charge[self.active_flower]=0
            
    
    
    def render_image(self, plot = False):
        self.image = np.zeros((self.N_vis,self.N_vis,3),dtype = float)
        
        self.image[:,:,:] = (self.background_color[None,None,:]/255.0)+np.random.rand(self.N_vis,self.N_vis,3)/20-0.025#np.random.rand(self.N_vis,self.N_vis)/10+0.7
        
        
    
    
    
        low_x = 0
        high_x = self.N_vis
    
        low_y = 0
        high_y = self.N_vis
    
    
        
    
        x, y = np.meshgrid(np.arange(self.N_vis,dtype = 'double')+0.5-self.N_vis/2,np.arange(self.N_vis,dtype = 'double')+0.5-self.N_vis/2)
    
        
        
        
        
        ro = 7.5#3.5
        
        
    
        
        '''
        valid_idx = np.linalg.norm(self.coord_list-self.cur_pos,axis = 1) < 2.0*self.R
        
        
        
        self.path = np.zeros(self.image.shape)
        
        for i in np.where(valid_idx)[0]:
            
            obj_x = self.coord_list[i,0]-self.cur_pos[0]
            obj_y = self.coord_list[i,1]-self.cur_pos[1]
            
            if (obj_x == 0) and (obj_y == 0):
                a = 1
                                                                
            rot_obj_x = np.cos(self.cur_orient+np.pi/2)*obj_x+np.sin(self.cur_orient+np.pi/2)*obj_y
            rot_obj_y = -np.sin(self.cur_orient+np.pi/2)*obj_x+np.cos(self.cur_orient+np.pi/2)*obj_y
            
            self.path[:,:,:2] += np.exp(-(np.sqrt((rot_obj_x-x[::-1].T)**2+(rot_obj_y-y[::-1].T)**2)/ro)**2)[:,:,None]
        
        self.path = self.path*3
        self.path[:,:,:2] -= 4.0
        
        self.path[:,:,:2] = (1.0/(1.0+np.exp(-self.path[:,:,:2])))
        
        #self.path[:,:,1] = 0.99
        
        
        #self.image = cv2.bitwise_and((self.image*255).astype(int),(self.path*255).astype(int),mask = None)/255
        
        
        
        
        self.image[:,:,:2] = self.image[:,:,:2]*(1.0-self.path[:,:,[0]])+self.path[:,:,:2]*(self.path[:,:,[0]])#np.maximum(self.path[:,:,:2]*(self.path[:,:,[0]]),self.image[:,:,[1]])
        '''
        
        
        
        
        # find the closest flower
        
        flw_dst = np.linalg.norm(self.honey_flowers_coord-self.cur_pos,axis = 1)
        closest_dist0 = flw_dst[0]
        closest_dist1 = flw_dst[1]
        
        vec0 = self.honey_flowers_coord[0]-self.cur_pos
        vec1 = self.honey_flowers_coord[1]-self.cur_pos
          
        flow_angle0 = np.arctan2(vec0[1],vec0[0])
        flow_angle1 = np.arctan2(vec1[1],vec1[0])
        
        rel_orient0 = -(self.cur_orient-flow_angle0)-np.pi/2
        rel_orient1 = -(self.cur_orient-flow_angle1)-np.pi/2
        
        
        red_halfW = self.halfW
        allblob_center0 = np.array([np.cos(rel_orient0+np.pi/2),np.sin(rel_orient0+np.pi/2)])*red_halfW#np.linalg.norm(full_coord)
        allblob_center1 = np.array([np.cos(rel_orient1+np.pi/2),np.sin(rel_orient1+np.pi/2)])*red_halfW#np.linalg.norm(full_coord)
        
        
        ####################
        
        
        max_visible_range = self.border_circle_rad*2
        min_visible_range = self.halfW/2.0
        
        min_poperek_inner = 10
        min_poperek_outer = 20
        
        max_poperek_inner = 30
        max_poperek_outer = 60
        
        min_vdol_inner = 10
        min_vdol_outer = 20
        
        max_vdol_inner = 30
        max_vdol_outer = 60
        
        N_rings = 5 # >= 2
        im_weight = 1.0/N_rings
        
        
        #image_copy = self.image.copy()
        
        if (closest_dist1<=min_visible_range):
            self.was_visited[1] = True
            
        if (closest_dist1 < max_visible_range) and (closest_dist1 > min_visible_range) and (not self.was_visited[1]):
        
            self.cum_image = np.zeros(self.image.shape, dtype = 'float')#self.image.copy()
            

            poperek_axis_inner = min_poperek_inner+((closest_dist1-min_visible_range)/\
                (max_visible_range-min_visible_range))**2*(max_poperek_inner-min_poperek_inner)
                
            poperek_axis_outer = min_poperek_outer+((closest_dist1-min_visible_range)/\
                (max_visible_range-min_visible_range))**2*(max_poperek_outer-min_poperek_outer)
                
                
            vdol_axis_inner = min_vdol_inner-((max_visible_range-closest_dist1)/\
                (max_visible_range-min_visible_range))**2*(-max_vdol_inner+min_vdol_inner)
            
            vdol_axis_outer = min_vdol_outer-((max_visible_range-closest_dist1)/\
                (max_visible_range-min_visible_range))**2*(-max_vdol_outer+min_vdol_outer)
            
            
            
            #rand_val = 1.0+(np.random.rand()-0.5)*2*0.03 #multiplitcative, +-3%
            #vdol_axis_inner *= rand_val
            #vdol_axis_outer *= rand_val
            
            #rand_val = 1.0+(np.random.rand()-0.5)*2*0.03 #multiplitcative, 3%
            #poperek_axis_inner *= rand_val
            #poperek_axis_outer *= rand_val
         
            
            for k in range(N_rings):
                vdol_axis = vdol_axis_inner+(vdol_axis_outer-vdol_axis_inner)*k/(N_rings-1.0)
                poperek_axis = poperek_axis_inner+(poperek_axis_outer-poperek_axis_inner)*k/(N_rings-1.0)
   
         
                self.inter_image= self.image.copy()
                
                axes = (int(round(poperek_axis)), int(round(vdol_axis)))
                angle = int(round(rel_orient1*180/np.pi))        
                startAngle = int(np.random.rand()*360)
                endAngle = int(startAngle+360)
    
                center = allblob_center1.copy().astype('int')
            # Generate ellipse points
                points = cv2.ellipse2Poly(center, axes, angle, startAngle, endAngle, delta=15).astype('double')
    
            
            
                
                #???self.flow_center_interp_points.append((np.vstack((x_new, y_new)).T))
                cur_color = np.array((1.0, 1.0, 0.0))
                cv2.fillPoly(self.inter_image, [np.array([points[:,0]+self.halfW,points[:,1]+self.halfW]).T.astype('int')], cur_color)
                self.cum_image += self.inter_image*im_weight
            self.image = self.cum_image
        
        
        if (closest_dist0<=min_visible_range):
            self.was_visited[0] = True
        if (closest_dist0 < max_visible_range) and (closest_dist0 > min_visible_range)and (not self.was_visited[0]):
        
            self.cum_image = np.zeros(self.image.shape, dtype = 'float')#self.image.copy()
            

            poperek_axis_inner = min_poperek_inner+((closest_dist0-min_visible_range)/\
                (max_visible_range-min_visible_range))**2*(max_poperek_inner-min_poperek_inner)
                
            poperek_axis_outer = min_poperek_outer+((closest_dist0-min_visible_range)/\
                (max_visible_range-min_visible_range))**2*(max_poperek_outer-min_poperek_outer)
                
                
            vdol_axis_inner = min_vdol_inner-((max_visible_range-closest_dist0)/\
                (max_visible_range-min_visible_range))**2*(-max_vdol_inner+min_vdol_inner)
            
            vdol_axis_outer = min_vdol_outer-((max_visible_range-closest_dist0)/\
                (max_visible_range-min_visible_range))**2*(-max_vdol_outer+min_vdol_outer)
            
            
            
            #rand_val = 1.0+(np.random.rand()-0.5)*2*0.03 #multiplitcative, +-3%
            #vdol_axis_inner *= rand_val
            #vdol_axis_outer *= rand_val
            
            #rand_val = 1.0+(np.random.rand()-0.5)*2*0.03 #multiplitcative, 3%
            #poperek_axis_inner *= rand_val
            #poperek_axis_outer *= rand_val
         
            
            for k in range(N_rings):
                vdol_axis = vdol_axis_inner+(vdol_axis_outer-vdol_axis_inner)*k/(N_rings-1.0)
                poperek_axis = poperek_axis_inner+(poperek_axis_outer-poperek_axis_inner)*k/(N_rings-1.0)
   
         
                self.inter_image= self.image.copy()
                
                axes = (int(round(poperek_axis)), int(round(vdol_axis)))
                angle = int(round(rel_orient0*180/np.pi))        
                startAngle = int(np.random.rand()*360)
                endAngle = int(startAngle+360)
    
                center = allblob_center0.copy().astype('int')
            # Generate ellipse points
                points = cv2.ellipse2Poly(center, axes, angle, startAngle, endAngle, delta=15).astype('double')
    
            
            
                
                #???self.flow_center_interp_points.append((np.vstack((x_new, y_new)).T))
                cur_color =np.array((0xcc, 0xbc, 0xf4),dtype = 'float')/255#np.array((0x87, 0xCE, 0xEB),dtype = 'float')/255 #np.array((0x9f, 0xd8, 0xfb),dtype = 'float')/255
                cv2.fillPoly(self.inter_image, [np.array([points[:,0]+self.halfW,points[:,1]+self.halfW]).T.astype('int')], cur_color)
                self.cum_image += self.inter_image*im_weight
            self.image = self.cum_image
            
            
        
            
            
            
        #self.image = self.image/2+image_copy/2
                
                
            
        valid_flower_idx = [0,1]
        self.add_flower(valid_flower_idx)
            
            
            
            
            
            
        
        self.image = np.clip(self.image,0.0,1.0)
        
        
        
                    
        #self.value_map = np.sum(np.exp(....))
        self.value = 0.0
        self.active_flower = None
        for v in valid_flower_idx:
            if np.linalg.norm(self.honey_flowers_coord[v]-self.cur_pos) < 20:
               if self.flowers_type[v] == 0:
                   self.value = self.violet_r*(self.violet_flower_charge[v]>1e-4)
               else:
                   self.value = self.red_r*(self.violet_flower_charge[v]>1e-4)
               self.active_flower = v
               break
            #self.value += (np.exp(-(np.linalg.norm(self.honey_flowers_coord[valid_flower_idx]-self.cur_pos,axis = 1)/10)**2)>0.5)
        dists = np.sqrt((x)**2+(y)**2)
        self.image[dists>self.R,:] = 0.95
        if plot:
            plt.figure()
            plt.imshow(self.image.transpose((1,0,2))[:,::-1,:], origin = 'lower')
            plt.xticks([])
            plt.yticks([])
                        
           
     
  
        
  
plt.close('all')
  
#JUMP3

#%% main pipeline



#reward (random)
#reward (init) 
# main_experiment
#reward 
#change values, retrain value
#reward
# main_experiment
#reward




bee = AI()
bee.load_state_dict(torch.load('C:/Users/Fedosov/Documents/projects/AI/weights_simple_improved_wide_critinit.pth'),strict = False)

#bee.world.violet_r = 1.0
#bee.world.red_r = 2.0
#bee.retrain_value()



print('START RANDOM BASELINE')
v1 = bee.plot_trajectories(N_trajectories = 2000, N_visible = 2000,random_action = True)
print('END RANDOM BASELINE')
print()

print('START INITIAL BASELINE')
v2 = bee.plot_trajectories(N_trajectories = 2000, N_visible = 2000)
print('END INITIAL BASELINE')
print()


print('part1')
bee.init_experiment()
#torch.save(bee.state_dict(), 'C:/Users/Fedosov/Documents/projects/AI/weights_1round_ver1.pth')
v3 = bee.plot_trajectories(N_trajectories = 2000, N_visible = 2000)

print('part2')
bee.world.violet_r = 1.0
bee.world.red_r = 2.0
bee.retrain_value()
#torch.save(bee.state_dict(), 'C:/Users/Fedosov/Documents/projects/AI/weights_2round_ver1.pth')
v4 =bee.plot_trajectories(N_trajectories = 2000, N_visible = 2000)

print('part3')
bee.init_experiment()
#torch.save(bee.state_dict(), 'C:/Users/Fedosov/Documents/projects/AI/weights_3round_ver1.pth')
v5 =bee.plot_trajectories(N_trajectories = 2000, N_visible = 2000)

value_array = np.array([v1,v2,v3,v4,v5])



plt.figure()
colorlist = ['black', 'gray','violet',[176/255,196/255,222/255],'red']
for i in range(5):
    plt.bar([i],value_array[i],color = colorlist[i])
    
plt.xlabel('stage')
plt.ylabel('mean reward per rollout')
plt.legend(['random','initial','trained violet max','values flipped','retrained red max'])





#np.save('values_ver1.npy',value_array)













'''
bee = AI()
bee.load_state_dict(torch.load('C:/Users/Fedosov/Documents/projects/AI/weights_simple_improved_wide_critinit.pth'),strict = False)

#bee.plot_trajectories(N_trajectories = 100, N_visible = 20,random_action = True)
#bee.plot_trajectories(N_trajectories = 100, N_visible = 20)

bee.init_experiment()
bee.plot_trajectories(N_trajectories = 100, N_visible = 20)

'''


'''
#%% init the critic
bee = AI()
bee.load_state_dict(torch.load('C:/Users/Fedosov/Documents/projects/AI/weights_simple_improved_wide_critinit.pth'),strict = False)

#bee.test_FE()
#bee.plot_actiohist()
#bee.plot_trajectories(N_trajectories = 10)
bee.plot_EFE()
bee.init_critic()
'''

  
        
'''
# torch.load('C:/Users/Fedosov/Documents/projects/AI/weights_simple_improved_wide.pth')
#basic weights

#%% init the action network

bee = AI()
bee.load_state_dict(torch.load('C:/Users/Fedosov/Documents/projects/AI/weights_simple_improved_wide_critinit.pth'),strict = False)
bee.test_FE()
bee.plot_actiohist()
bee.init_actionet()
bee.plot_actiohist()




'''





'''

#%% initial random action world learning
bee = AI()
bee.load_state_dict(torch.load('C:/Users/Fedosov/Documents/projects/AI/weights_simple_improved.pth'),strict = False)
bee.forced_train_FE_separate()
#NEVER UNCOMMENT NEXT LINE!!!
#torch.save(bee.state_dict(), 'C:/Users/Fedosov/Documents/projects/AI/weights_simple.pth')

'''













#%%


#Visualization
plt.close('all')
bee.world.generate_world()
bee.world.init_random_pos()
bee.world.cur_pos = bee.world.honey_flowers_coord[0,:]
bee.world.render_image(plot = True)
bee.world.render_image(plot = True)
plt.xticks([]) 
plt.yticks([])





plt.close('all')
bee.world.generate_world()
bee.world.init_random_pos()
bee.world.cur_pos = bee.world.honey_flowers_coord[1,:]
bee.world.render_image(plot = True)
bee.world.render_image(plot = True)
plt.xticks([]) 
plt.yticks([])



























#%%

np.random.seed(0)

plt.close('all')
from matplotlib import patches

plt.close('all')
bee.world.generate_world()
bee.world.init_random_pos()
bee.world.render_image(plot = True)
plt.xticks([]) 
plt.yticks([])



fig,ax = plt.subplots()
circle1 = patches.Circle((0.0, 0.0), radius=120,linestyle = '--',color='yellow',fill = False)
ax.add_patch(circle1)



ax.add_patch(circle1)

ax.set_xlim(-130,130)
ax.set_ylim(-130,130)
ax.set_aspect(1.0)


circle2 = patches.Circle((0.0, 0.0), radius=50,linestyle = '-',color='lightblue',fill = True)
ax.add_patch(circle2)


rot = np.array([[np.cos(bee.world.cur_orient),+np.sin(bee.world.cur_orient)],
                [-np.sin(bee.world.cur_orient),np.cos(bee.world.cur_orient)]])

triangle = patches.Polygon([np.array((2, 0))@rot*10, np.array((-1, 1))@rot*10, \
                            np.array((-1, -1))@rot*10], closed=True, edgecolor='orange', facecolor='orange')

ax.add_patch(triangle)
    

for i in range(1):
    #angle = np.random.rand()*2*np.pi/3+2*np.pi/3#np.linspace(2*np.pi/3,np.pi+np.pi/3):
    circle = patches.Circle((np.cos(bee.world.rem_flow_ori0)*100, np.sin(bee.world.rem_flow_ori0)*100), 
                            radius=20,linestyle = '-',color='violet',fill = True)
    ax.add_patch(circle)
    circle = patches.Circle((np.cos(bee.world.rem_flow_ori0)*100, np.sin(bee.world.rem_flow_ori0)*100), 
                            radius=30,linestyle = 'dotted',color='violet',fill = False,linewidth = 2)
    ax.add_patch(circle)
    
    
    angle = np.random.rand()*2*np.pi/3-1*np.pi/3#np.linspace(2*np.pi/3,np.pi+np.pi/3):
    circle = patches.Circle((np.cos(bee.world.rem_flow_ori1)*100, np.sin(bee.world.rem_flow_ori1)*100), 
                            radius=20,linestyle = '-',color='red',fill = True)
    ax.add_patch(circle)
    circle = patches.Circle((np.cos(bee.world.rem_flow_ori1)*100, np.sin(bee.world.rem_flow_ori1)*100),
                            radius=30,linestyle = 'dotted',color='red',fill = False, linewidth = 2)
    ax.add_patch(circle)
        
    
    
for i in range(20):
    angle = np.random.rand()*2*np.pi/3+2*np.pi/3#np.linspace(2*np.pi/3,np.pi+np.pi/3):
    circle = patches.Circle((np.cos(angle)*100, np.sin(angle)*100), 
                            radius=3,linestyle = '-',color='violet',fill = True, alpha = 0.5)
    ax.add_patch(circle)
    
    
    angle = np.random.rand()*2*np.pi/3-1*np.pi/3#np.linspace(2*np.pi/3,np.pi+np.pi/3):
    circle = patches.Circle((np.cos(angle)*100, np.sin(angle)*100), 
                            radius=3,linestyle = '-',color='red',fill = True, alpha = 0.5)
 
    ax.add_patch(circle)
            
    
    
    
    

ax.set_xticks([]) 
ax.set_yticks([])





#good path

'''
moves_list = [[0.5,0.5],[1.0,-np.pi/2],[0.8,-0.1],[0.2,0.2],[0.2,0.8]]

for move in moves_list:
    bee.world.action_step(move[0], move[1])
    bee.world.render_image(plot = True)
    plt.xticks([]) 
    plt.yticks([])
'''

moves_list = [[0.3,0.9],[0.3,0.8],[0.8,-0.1],[1.0,-0.5],[1.0,0.0]]

for move in moves_list:
    bee.world.action_step(move[0], move[1])
    bee.world.render_image(plot = True)
    plt.xticks([]) 
    plt.yticks([])
    
    
    











#%
np.random.seed(0)

plt.close('all')

bee.world.generate_world()
bee.world.init_random_pos()
bee.world.render_image(plot = True)

moves =  [[0.5,0.7],[1.0,-np.pi/2],[0.8,-0.1],[0.5,0.2],[0.2,0.8],[0.2,0.0]]

bee.test_FE(moves)














#%
np.random.seed(2)
torch.manual_seed(5)

bee =AI()
#bee.load_state_dict(torch.load('C:/Users/Fedosov/Documents/projects/AI/weights_simple_improved_wide_FULL.pth'),strict = False)


bee.load_state_dict(torch.load( 'C:/Users/Fedosov/Documents/projects/AI/weights_1round_ver1.pth'),strict = False)

plt.close('all')

bee.world.generate_world()
bee.world.init_random_pos()
bee.plot_EFE()















#%%
plt.close('all')
np.random.seed(2)
torch.manual_seed(0)

bee =AI()
#bee.load_state_dict(torch.load('C:/Users/Fedosov/Documents/projects/AI/weights_simple_improved_wide_FULL.pth'),strict = False)


bee.load_state_dict(torch.load( 'C:/Users/Fedosov/Documents/projects/AI/weights_3round_ver1.pth'),strict = False)



bee.plot_trajectories(100,100, random_action = True)
















