import numpy as np
import scipy,psutil,shelve,sparse
import tensorflow as tf
from time import time
from scipy.sparse import diags, bmat, identity,lil_array,csr_array,block_diag
from scipy.sparse.linalg import eigs,spsolve,svds,lsmr
import matplotlib.pyplot as plt
from copy import deepcopy
from statsmodels.nonparametric.smoothers_lowess import lowess

class spline_break:
    def __init__(self,parameter,guess=None,full_dist=False):
        parameter = parameter / parameter[4]
        self.guess = None
        self.koff = parameter[0]
        self.kon = parameter[1]
        self.f = parameter[2]
        self.h = parameter[3]
        self.dm = parameter[4]
        self.kp = parameter[5]
        self.dp = parameter[6]
        self.max_state = max(self.kon, self.koff)
        self.protein_range = int(self.max_state * self.kp / self.dp)
        self.protein_range=self.protein_range+int(self.protein_range**0.5*20)
        self.max_state = int(self.max_state + self.max_state ** 0.5 * 4)
        self.protein_break=np.hstack((np.arange(int(self.protein_range/30)),np.arange(int(self.protein_range/30),int(self.protein_range/20),max(int(self.kp/self.dp/20),4)),np.arange(int(self.protein_range/20),int(self.protein_range/10),max(int(self.kp/self.dp/15)),6),np.arange(int(self.protein_range/10),int(self.protein_range/6),max(int(self.kp/self.dp/4),20))))
        #self.protein_break = np.hstack((self.protein_break, np.arange(self.protein_break[-1], int(self.protein_range/3),int(self.kp / self.dp/4))))
        self.protein_break=np.hstack((self.protein_break,np.linspace(self.protein_break[-1],self.protein_range-int(self.protein_range/10),20).astype('int')))
        self.protein_break=np.hstack((self.protein_break,np.arange(self.protein_break[-1],self.protein_range,int(self.kp / self.dp/4)),self.protein_range-1))
        self.protein_break = list(set(self.protein_break))
        self.protein_break.sort()
        self.protein_break = np.array(self.protein_break)
        self.partition_end = np.hstack((self.protein_break[1:] - 1, self.protein_break[-1]))
        dkd = diags([-np.arange(self.max_state), np.arange(1, self.max_state)], offsets=[0, 1])
        ones = np.ones(self.max_state)
        ones[-1] = 0
        tridiag = diags([-ones, ones], offsets=[0, -1])
        self.A = tridiag * self.koff + dkd
        self.B = tridiag * self.kon + dkd
        self.ident = identity(self.max_state)
        self.transition = bmat([[self.A, self.f * self.ident], [None, self.B - self.f * self.ident]])
        self.protein_translate = diags([np.hstack((np.arange(self.max_state), np.arange(self.max_state)))],offsets=[0]) * self.kp
        self.T = block_diag([self.transition for i in range(self.protein_break.shape[0])])
        self.H=lil_array(self.T.shape)
        self.degrade_block=lil_array(self.T.shape)
        self.translate_block=lil_array(self.T.shape)
        self.mean = np.zeros((3, self.protein_break.shape[0]))
        self.gene_mean = np.zeros((2, self.protein_break.shape[0],2))
        self.moment = np.zeros((2, self.protein_break.shape[0], 2))
        self.var = np.zeros((2, self.protein_break.shape[0], 2))
        if guess is not None:
            self.G = guess.sum(axis=2)[:, 0] / guess.sum(axis=(1, 2))
            if full_dist:
                self.weight = guess.sum(axis=(1, 2))
                self.P = deepcopy(guess)
                mean=(self.P/self.P.sum(axis=-1)[:,:,None]*np.arange(self.max_state)[None,None,:]).sum(axis=-1)
                var=(self.P/self.P.sum(axis=-1)[:,:,None]*np.arange(self.max_state)[None,None,:]**2).sum(axis=-1)-mean**2
                self.mean_active=mean[:,1]
                self.mean_inactive=mean[:,0]
                self.var_active=var[:,1]
                self.var_inactive=var[:,0]
                self.P = np.add.reduceat(guess, self.protein_break, axis=0)
                self.P = np.tile(self.P, (3, 1, 1, 1))
                self.P[1]=deepcopy(guess[self.protein_break])
                self.P[-1]=deepcopy(guess[self.partition_end])
            else:
                try:
                    self.P = np.add.reduceat(guess, self.protein_break, axis=0)
                except:
                    self.P=deepcopy(guess)
                self.P = np.tile(self.P, (3, 1, 1, 1))
                self.weight=np.ones(self.protein_range)
                self.mean_inactive, self.mean_active, self.var_inactive, self.var_active = self.interpolate_mean_var()
            guess=True

        else:
            self.weight = np.ones(self.protein_range)
            self.G = np.ones(self.protein_range)
            self.P = np.ones((self.protein_break.shape[0], 2, self.max_state))
            self.P=self.P/self.P.sum()
            self.P = np.tile(self.P, (3, 1, 1, 1))
            guess=False
        for index, i in enumerate(self.protein_break[:-1]):
            if self.protein_break[index + 1] - self.protein_break[index] > 1:
                self.weight[self.protein_break[index]:self.protein_break[index + 1]] = self.weight[self.protein_break[index]:self.protein_break[index + 1]] / self.weight[self.protein_break[index]:self.protein_break[index + 1]].sum()
            else:
                self.weight[i] = 1
        self.weight[-1] = 1
        #self.spine_solve(guess=guess)
        self.iterative(guess=guess,plot=False)

    def interpolate_mean_var(self,plot=False):
        total_P=self.P[0,:].sum(axis=(-1,-2))
        overall_normalized=self.P[0,:].sum(axis=1)/total_P[:,None]
        gene_normalized=self.P[0,:]/self.P[0,:].sum(axis=-1)[:,:,None]
        total_G=np.tile(self.P[0,:,0,:].sum(axis=-1)/self.P[0,:].sum(axis=(-1,-2)),(3,1))
        overall_mean = np.tile((overall_normalized * np.arange(self.max_state)[None,:]).sum(axis=-1),(3,1))
        gene_mean=np.tile((gene_normalized*np.arange(self.max_state)[None,None,:]).sum(axis=-1),(3,1,1))
        gene_second_moment=np.tile((gene_normalized*np.arange(self.max_state)[None,None,:]**2).sum(axis=-1),(3,1,1))
        gene_third_moment=np.tile((gene_normalized*np.arange(self.max_state)[None,None,:]**3).sum(axis=-1),(3,1,1))
        gene_var=gene_second_moment[0]-gene_mean[0]**2
        if plot:
            #plt.plot((test2.v.sum(axis=1)/test2.v.sum(axis=(1,2))[:,None]*np.arange(30)).sum(axis=-1))
            plt.scatter(self.protein_break,overall_mean[0])
            plt.show()
        for index, i in enumerate(self.protein_break[:-1]):
            start = self.protein_break[index]
            end = self.protein_break[index + 1]
            if end-start==1:
                continue
            b = np.zeros(end - start)
            b[-1] = 1
            tmp=self.weight[start:end]
            tmp=tmp/tmp.sum()
            weight_tmp=np.zeros(end-start)
            counter=0
            while np.linalg.norm(weight_tmp-tmp)>0.001:
                if counter:
                    weight_tmp=(tmp+weight_tmp)/2
                else:
                    weight_tmp=tmp
                weight_tmp=weight_tmp/weight_tmp.sum()
                protein_avg=(np.arange(start,end)*weight_tmp).sum()
                slope=(overall_mean[0,index]-overall_mean[-1,index-1])/(protein_avg-start+1)
                if slope<0:
                    break
                trans_diags = overall_mean[-1,index-1]+slope * np.arange(1,end - start+1)
                translate=diags([trans_diags*self.kp,-np.arange(start+1,end)*self.dp],offsets=[0,1],shape=(end-start,end-start)).tolil()
                translate[-1]=trans_diags*self.kp
                b[-1]=overall_mean[0,index]
                tmp=spsolve(translate.tocsr(),b)
                tmp=tmp/tmp.sum()
                counter+=1
                if counter>100 or (tmp<0).any():
                    tmp = self.weight[start:end]
                    break
            if slope>0:
                overall_mean[-1,index]=trans_diags[-1]
                overall_mean[1,index]=trans_diags[0]
                avg=np.mean(tmp[1:]/tmp[:-1])
                if avg>1.15:
                    tmp=1.15**np.arange(start,end)
                elif avg<0.9:
                    tmp=0.9**np.arange(start,end)
                tmp = tmp / tmp.sum()
                self.weight[start:end] = tmp


            c1=(gene_second_moment[0,index,:]-gene_second_moment[-1,index-1,:])/(tmp*np.arange(1,end-start+1)).sum()
            tmp_moment1=(gene_second_moment[-1,index-1,:]+c1*np.arange(1,end-start+1)[:,None])
            gene_second_moment[-1,index,:]=tmp_moment1[-1,:]
            gene_second_moment[1,index,:]=tmp_moment1[0,:]

            c2 = (gene_third_moment[0, index,:] - gene_third_moment[-1, index-1,:])/(tmp * np.arange(1, end - start + 1)).sum()
            tmp_moment2 = (gene_third_moment[-1, index,:] + c2 * np.arange(1, end - start + 1)[:, None])
            gene_third_moment[-1,index,:]=tmp_moment2[-1,:]
            gene_third_moment[1,index,:]=tmp_moment2[0,:]

            c3=(total_G[0,index]-total_G[0,index-1])/(tmp*np.arange(1,end-start+1)).sum()
            tmp_G=total_G[0,index-1]+c3*np.arange(1,end-start+1)
            total_G[1,index]=tmp_G[0]
            total_G[-1,index]=tmp_G[-1]


            gene_mean[1,index,0]=-(overall_mean[0,index]-gene_mean[0,index,0])+overall_mean[1,index]
            gene_mean[1, index, 1] = (gene_mean[0, index,1] - overall_mean[0,index]) + overall_mean[1,index]

        #gene_mean[1,:,0]= lowess(gene_mean[1,:,0], self.protein_break, frac=0.1)[:,1]
        #gene_mean[1,:,1] = lowess(gene_mean[1,:,1], self.protein_break, frac=0.1)[:,1]
        gene_mean_inter0=scipy.interpolate.interp1d(self.protein_break,gene_mean[1,:,0],kind='cubic')
        gene_mean_approx0=gene_mean_inter0(np.arange(self.protein_range))
        gene_mean_inter1=scipy.interpolate.interp1d(self.protein_break,gene_mean[1,:,1],kind='cubic')
        gene_mean_approx1=gene_mean_inter1(np.arange(self.protein_range))

        """
        for index,i in enumerate(self.protein_break[:-1]):
            start=self.protein_break[index]
            end=self.protein_break[index+1]
            if end-start==1:
                continue
            gene_var[]
        """
        gene_var[:,0]=lowess(gene_var[:,0], self.protein_break, frac=0.1)[:,1]
        gene_var[:,1] = lowess(gene_var[:,1], self.protein_break, frac=0.1)[:,1]
        gene_var_inter0=scipy.interpolate.interp1d(self.protein_break,gene_var[:,0],kind='cubic')
        gene_var_approx0=gene_var_inter0(np.arange(self.protein_range))
        gene_var_inter1=scipy.interpolate.interp1d(self.protein_break,gene_var[:,1],kind='cubic')
        gene_var_approx1=gene_var_inter1(np.arange(self.protein_range))


        G_inter = scipy.interpolate.interp1d(self.protein_break, total_G[1], kind='cubic')
        G= G_inter(np.arange(self.protein_range))
        if plot:
            plt.scatter(np.arange(self.protein_range),gene_mean_approx0)
            plt.scatter(np.arange(self.protein_range), gene_mean_approx1)
            #plt.plot((test2.v / test2.v.sum(axis=( 2))[:,:, None] * np.arange(30)[None,None :]).sum(axis=-1))
            plt.show()

            plt.scatter(np.arange(self.protein_range),gene_var_approx0)
            plt.scatter(np.arange(self.protein_range), gene_var_approx1)
            #plt.plot((test2.v / test2.v.sum(axis=( 2))[:,:, None] * np.arange(30)[None,None :]**2).sum(axis=-1)-(test2.v / test2.v.sum(axis=( 2))[:,:, None] * np.arange(30)[None,None :]).sum(axis=-1)**2)
            plt.show()
        if (gene_var_approx0<0).any():
            gene_var_approx0=self.var_inactive
        else:
            try:
                gene_var_approx0=self.var_inactive/2+gene_var_approx0/2
            except:
                pass
        if (gene_var_approx1 < 0).any():
            gene_var_approx1=self.var_active
        else:
            try:
                gene_var_approx1 = self.var_active /2 +gene_var_approx1/2
            except:
                pass
        for index,i in enumerate(self.protein_break):
            start=self.protein_break[index]
            end=self.partition_end[index]
            if start==end:
                continue
            #if gene_mean_approx0[start]>self.koff
            tmp=scipy.stats.norm.pdf(np.arange(self.max_state),loc=gene_mean_approx0[start],scale=gene_var_approx0[start]**0.5)
            tmp=tmp/tmp.sum()
            self.P[1,index,0]=tmp*total_P[index]*self.weight[start]*G[start]

            tmp=scipy.stats.norm.pdf(np.arange(self.max_state),loc=gene_mean_approx0[end],scale=gene_var_approx0[end]**0.5)
            tmp=tmp/tmp.sum()
            self.P[-1,index,0]=tmp*total_P[index]*self.weight[end]*G[end]

            tmp=scipy.stats.norm.pdf(np.arange(self.max_state),loc=gene_mean_approx1[start],scale=gene_var_approx1[start]**0.5)
            tmp=tmp/tmp.sum()
            self.P[1,index,1]=tmp*total_P[index]*self.weight[start]*(1-G[start])

            tmp=scipy.stats.norm.pdf(np.arange(self.max_state),loc=gene_mean_approx1[end],scale=gene_var_approx1[end]**0.5)
            tmp=tmp/tmp.sum()
            self.P[-1,index,1]=tmp*total_P[index]*self.weight[end]*(1-G[end])


        return [gene_mean_approx0,gene_mean_approx1,gene_var_approx0,gene_var_approx1]

    def construct_block(self,first=False,guess=True,plot=False):
        if first and guess:
            ratio=1
        elif first:
            ratio=1
        elif guess:
            ratio=0.2
        block_size=2*self.max_state
        for i in range(self.protein_break.shape[0]):
            start=self.protein_break[i]
            end=self.partition_end[i]
            if (end==start and first) or not guess:
                self.H[i*block_size:(i+1)*block_size,i*block_size:(i+1)*block_size]=(start+end)/2*self.h*bmat([[-self.ident,None],[self.ident,np.zeros((self.max_state,self.max_state))]])
                if i!=0:
                    self.degrade_block[i*block_size:(i+1)*block_size,i*block_size:(i+1)*block_size]=-identity(2 * self.max_state) * self.dp * start*self.weight[start]
                    self.degrade_block[(i-1)*block_size:i*block_size,i*block_size:(i+1)*block_size]=identity(2 * self.max_state) * self.dp * start*self.weight[start]
                if i!=self.protein_break.shape[0]-1:
                    self.translate_block[i*block_size:(i+1)*block_size,i*block_size:(i+1)*block_size]=-self.protein_translate*self.weight[end]
                    self.translate_block[(i+1)*block_size:(i+2)*block_size,i*block_size:(i+1)*block_size]=self.protein_translate*self.weight[end]
            elif end !=start:
                total_weight=(np.arange(start,end+1)*self.weight[start:end+1]).sum()
                mean=(self.mean_inactive[start:end+1]*self.weight[start:end+1]*np.arange(start,end+1)).sum()/total_weight
                var=((self.var_inactive[start:end+1]+(self.mean_inactive[start:end+1]-mean)**2)*self.weight[start:end+1]*np.arange(start,end+1)).sum()/total_weight
                Effective_H_vec=scipy.stats.norm.pdf(np.arange(30),loc=mean,scale=var**0.5)
                Effective_H_vec=Effective_H_vec/Effective_H_vec.sum()*total_weight*self.P[0,i,0].sum()*self.h

                h=diags([Effective_H_vec/self.P[0,i,0]],offsets=[0])
                h=bmat([[-h,None],[h,np.zeros((self.max_state,self.max_state))]])
                self.H[i*block_size:(i+1)*block_size,i*block_size:(i+1)*block_size]=self.H[i*block_size:(i+1)*block_size,i*block_size:(i+1)*block_size]*(1-ratio)+h*ratio
                degrade=diags([self.P[1,i].ravel()/self.P[0,i].ravel()],offsets=[0])*self.dp*start
                self.degrade_block[(i-1)*block_size:i*block_size,i*block_size:(i+1)*block_size]=degrade
                self.degrade_block[i*block_size:(i+1)*block_size,i*block_size:(i+1)*block_size]=-degrade
                translate = self.protein_translate @ diags([self.P[-1, i].ravel() / self.P[0, i].ravel()], offsets=[0])
                self.translate_block[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size] = (self.translate_block[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size])*(1-ratio)-translate*ratio
                self.translate_block[(i+1) * block_size:(i + 2) * block_size, i * block_size:(i + 1) * block_size] = translate*ratio+(self.translate_block[(i+1) * block_size:(i + 2) * block_size, i * block_size:(i + 1) * block_size])*(1-ratio)
                if plot:
                    plt.plot(Effective_H_vec)
                    #truth=(test2.v[start:end+1,0]*np.arange(start,end+1)[:,None]*self.h).sum(axis=0)
                    #plt.plot(truth)
                    plt.show()
                    plt.plot((degrade@self.P[0,i].ravel()).reshape(2,30).T)
                    #plt.plot(test2.v[start].T*self.dp*start)
                    plt.show()
                    plt.plot((translate@self.P[0,i].ravel()).reshape(2,30).T)
                    #plt.plot((self.protein_translate@test2.v[end].ravel()).reshape(2,30).T)
                    plt.title('{}:{} translate'.format(self.protein_break[i], self.partition_end[i]))
                    plt.show()
                    print('done')

    def iterative(self,guess=False,plot=False):
        self.construct_block(first=True,guess=guess,plot=True)
        error=10
        #if not guess:
            #self.inactive_mean, self.active_mean, self.inactive_var, self.active_var = self.interpolate_mean_var()
        while error>10**-2:
            self.P_store=deepcopy(self.P)
            try:
                E,V=eigs(self.T+self.H+self.degrade_block+self.translate_block,k=1,sigma=10**-11)
                V=np.abs(np.real(V)).reshape(self.protein_break.shape[0],2,self.max_state)
                self.P[0]=V/V.sum()#/2*0.5+self.P[0]*0.5
            except:
                print('singular')
            error=np.abs(self.P[0].sum(axis=0)-self.P_store[0].sum(axis=0)).sum()
            if plot:
                plt.plot(self.P[0].sum(axis=0).T)
                plt.plot(self.P_store[0].sum(axis=0).T)
                plt.show()
            self.mean_inactive,self.mean_active,self.var_inactive,self.var_active=self.interpolate_mean_var(plot=False)
            self.construct_block(plot=False)#,guess=True)

    def spine_solve(self,guess=False,plot=False):
        self.construct_block(first=True,guess=guess)
        error=10
        if not guess:
            self.inactive_mean, self.active_mean, self.inactive_var, self.active_var = self.interpolate_mean_var()
        while error>10**-3:
            self.P_store=deepcopy(self.P)
            """
            for index,i in enumerate(self.protein_break):
                start=self.protein_break[index]
                end=self.partition_end[index]
                if index==0:
                    tmp=spsolve(self.transition+self.H[index*60:(index+1)*60,index*60:(index+1)*60]-self.protein_translate,-self.P[1,index+1].ravel()*self.dp)
                elif start==self.protein_break[-1]:
                    tmp=spsolve(self.transition+self.H[index*60:(index+1)*60,index*60:(index+1)*60]-self.dp*self.protein_break[-1]*identity(60),-self.protein_translate@self.P[-1,index-1].ravel())
                else:
                    tmp=spsolve(self.transition+(self.degrade_block+self.translate_block+self.H)[index*60:(index+1)*60,index*60:(index+1)*60],-self.dp*(end+1)*self.P[1,index+1].ravel()-self.protein_translate@self.P[-1,index-1].ravel())
                self.P[0, index] = self.P[0, index]*0 + tmp.reshape(2, self.max_state)*1
                if plot:
                    plt.plot((tmp.reshape(2,30)+self.P[0,index]).T/2)
                    plt.plot(self.P[0,index].T)
                    plt.scatter(np.arange(30),test2.v[start:end+1,0].sum(axis=0))
                    plt.scatter(np.arange(30), test2.v[start:end+1, 1].sum(axis=0))
                    plt.title(self.protein_break[index])
                    plt.show()
                    dummy=0
            """
            P_diff=((self.T+self.H+self.degrade_block+self.translate_block)@self.P[0].ravel()).reshape(self.protein_break.shape[0], 2, self.max_state)
            plt.plot(P_diff.sum(axis=0).T)
            plt.show()
            self.P[0]+=P_diff*0.1
            plt.plot(self.P[0].sum(axis=0).T)
            plt.show()
            #"""
            self.mean_inactive, self.mean_active, self.var_inactive, self.var_active = self.interpolate_mean_var()
            self.construct_block()
            error=np.abs(self.P_store[0]-self.P[0]).sum()


class construct_full_cme:
    # surrogate model of g*+p<->g+p
    # construct by triangular block at different protein number
    def __init__(self,param,guess=None):
        start=time()
        self.param=param/param[4]
        self.mrna_range=int(self.param[1]+4*self.param[1]**0.5)
        self.protein_range=int(self.param[1]*self.param[-2]/self.param[-1])
        self.protein_range=self.protein_range+int(self.protein_range**0.5*20)
        self.full_matrix = []
        protein_production = diags([np.hstack((np.arange(self.mrna_range) * self.param[-2], np.arange(self.mrna_range) * self.param[-2]))],offsets=[0])
        protein_degradation = np.eye(self.mrna_range*2)*param[-1]
        one = np.ones(self.mrna_range)
        one[-1] = 0
        ones=np.eye(self.mrna_range)
        self.mrna_production = diags([-one ,one[:-1] ], offsets=[0,-1])
        self.mrna_degradation = diags([-np.arange(self.mrna_range), np.arange(1,self.mrna_range)],offsets=[0,1])
        for i in range(self.protein_range):
            mrna_block=bmat([[self.mrna_production*self.param[0]+self.mrna_degradation-ones*i*param[3],None],[None,self.mrna_production*self.param[1]+self.mrna_degradation-ones*param[2]]])+bmat([[None,ones*param[2]],[ones*i*param[3],None]])
            self.full_matrix.append(([None]*(i-1) if i>1 else [])+([protein_production] if i>0 else [])+[mrna_block-(protein_production if i<self.protein_range-1 else 0)-protein_degradation*i]+([protein_degradation*(i+1)] if (i+1)<self.protein_range else [])+[None]*(self.protein_range-i-2))
        full_matrix=bmat(self.full_matrix)
        start2=time()

        if guess is None:
            self.v=np.abs(np.real(scipy.sparse.linalg.eigs(full_matrix,k=1,sigma=0)[1]))
        else:
            self.v = np.abs(np.real(scipy.sparse.linalg.eigs(full_matrix, k=1, sigma=0,v0=guess.ravel())[1]))
        print(time()-start2,start2-start)
        self.v=self.v/self.v.sum()
        self.v=self.v.reshape(self.protein_range,2,self.mrna_range)

if __name__=='__main__':

    param=np.array([[1,15,0.08,0.001,1,30,1],[1,15,0.08,0.08,1,8,1],[1,15,0.08,0.0008,1,8,1],[1,15,0.08,0.001,1,20,1]])
    start=time()
    """
    fig,ax=plt.subplots(2,2,figsize=(10,10))
    counter=0
    for i in range(param.shape[0]):
        test=construct_full_cme_3_(param[i,:])
        test3=spline_break5(param[i,:],guess=test.v,full_dist=True)
        ax[counter//2,counter%2].plot(test.v.sum(axis=2))
        ax[counter//2,counter%2].scatter(np.arange(30),test3.P.sum(axis=2)[:,0])
        ax[counter // 2, counter % 2].scatter(np.arange(30),test3.P.sum(axis=2)[:, 1])
        counter+=1
    fig.tight_layout()
    """
    #test=construct_full_cme_3_(param[0])
    #test2 = construct_full_cme_3_(np.array([1,15,0.08,0.008,1,8,1]))
    print(time()-start,flush=True)
    test2=construct_full_cme(param[0])
    #test3=spline_break5(param[0],guess=test.v,full_dist=True)
    test2_2 = construct_full_cme(np.array([1,15,0.08,0.0008,1,8,1]))
    start=time()
    test3=spline_break(param[0,:],guess=test2.v,full_dist=False)
    print(time()-start)