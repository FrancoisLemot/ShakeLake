# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:30:18 2024

@author: lemotf
"""

#%% CODE library import

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

#%% Faults definition

#The faults objects are defined listes of point coordinates [easting, northing] forming a polygonal line.
#Two consecutive points define the elementary segment that can break.
#Simulated ruptures include all continuous combination of segments within a fault.
#Ruptures cnnat "jump" from one fault to another.

#define faults paths vertices
F1=[[	626777	,	3499829	]	,
[	630366	,	3496734	]	,
[	634767	,	3494099	]	,
[	638107	,	3491957	]	,
[	641798	,	3488867	]	,
[	644311	,	3486207	]	,
[	646591	,	3483742	]	,
[	649042	,	3481152	]	,
[	651645	,	3478544	]	,
[	653545	,	3476464	]	,
[	655797	,	3474150	]	,
[	658035	,	3472131	]	,
[	660409	,	3469691	]	,
[	662594	,	3467350	]	,
[	664925	,	3465085	]	,
[	667364	,	3462872	]	,
[	669622	,	3460716	]	,
[	671850	,	3458328	]	,
[	674578	,	3455418	]	,
[	677276	,	3453276	]	,
[	680100	,	3451106	]	,
[	682063	,	3448974	]	,
[	684569	,	3446366	]	,
[	687277	,	3443864	]	,
[	689361	,	3441575	]	,
[	691904	,	3438847	]	,
[	694442	,	3436213	]	,
[	697289	,	3433259	]	,
[	699866	,	3430640	]	,
[	701900	,	3428483	]	,
[	704618	,	3426140	]	,
[	707122	,	3424013	]	,
[	709804	,	3421318	]	,
[	711928	,	3419114	]	,
[	714096	,	3416585	]	,
[	716113	,	3413936	]	,
[	717768	,	3411768	]	,
[	719334	,	3409902	]	,
[	721336	,	3407713	]	,
[	723507	,	3404986	]	,
[	725244	,	3402944	]	,
[	726433	,	3400400	]	,
[	728255	,	3397662	]	,
[	729825	,	3395323	]	,
[	731712	,	3392392	]	,
[	733299	,	3390152	]	,
[	734792	,	3388098	]	,
[	736509	,	3385772	]	,
[	738195	,	3383725	]	,
[	739961	,	3381988	]	,
[	743228	,	3381554	]	,
[	746118	,	3381449	]	,
[	748567	,	3380102	]	,
[	750864	,	3379288	]	,
[	752985	,	3377317	]	,
[	754920	,	3375757	]	,
[	756993	,	3374262	]	,
[	758719	,	3372230	]	,
[	760154	,	3369596	]	,
[	762195	,	3366787	]	,
[	764657	,	3364026	]	,
[	766290	,	3362237	]	,
[	768628	,	3359473	]	,
[	770911	,	3356992	]	,
[	772689	,	3354358	]	,
[	775904	,	3351887	]	,
[	777901	,	3348807	]	,
[	779102	,	3345652	]	,
[	780801	,	3342267	]	,
[	781865	,	3338489	]	]


F3=[[	762230	,	3341923	]	,
[	764164	,	3340507	]	,
[	765581	,	3338051	]	,
[	766547	,	3336144	]	,
[	767957	,	3334256	]	,
[	769310	,	3331842	]	,
[	771295	,	3329244	]	,
[	772849	,	3326840	]	,
[	774927	,	3323981	]	,
[	776453	,	3321972	]	,
[	779041	,	3320621	]	,
[	781438	,	3319843	]	]

F2=[[	741333	,	3379947	]	,
[	742558	,	3377228	]	,
[	743226	,	3374844	]	,
[	744515	,	3370814	]	,
[	745540	,	3368682	]	,
[	746613	,	3366351	]	,
[	748166	,	3363625	]	,
[	749786	,	3361679	]	,
[	752284	,	3359265	]	,
[	755670	,	3357099	]	,
[	757321	,	3354534	]	,
[	759126	,	3352081	]	,
[	760412	,	3349876	]	,
[	762695	,	3347408	]	,
[	764587	,	3345798	]	,
[	766113	,	3344003	]	,
[	768103	,	3342082	]	,
[	770343	,	3340635	]	,
[	772481	,	3339502	]	,
[	774843	,	3338331	]	,
[	776685	,	3336477	]	,
[	779073	,	3334382	]	,
[	780575	,	3331812	]	,
[	781775	,	3329470	]	,
[	783570	,	3326832	]	,
[	785023	,	3324115	]	,
[	785925	,	3321028	]	,
[	787174	,	3317272	]	,
[	788294	,	3314109	]	,
[	789542	,	3311817	]	,
[	791165	,	3308909	]	,
[	792430	,	3306305	]	,
[	793936	,	3303701	]	,
[	794970	,	3300325	]	,
[	795859	,	3296555	]	,
[	796899	,	3293442	]	,
[	798178	,	3290226	]	,
[	799455	,	3286959	]	,
[	801259	,	3282968	]	,
[	802348	,	3279675	]	,
[	803260	,	3276186	]	,
[	804172	,	3272221	]	,
[	805382	,	3267324	]	,
[	807159	,	3262617	]	,
[	809903	,	3257208	]	]

#Convert in km
F1=np.array(F1)/1000
F2=np.array(F2)/1000
F3=np.array(F3)/1000
Faults=[F1,F2,F3] #Fault network to explore for ruptures

#%% INPUT sites location

#provide lake coordinates
Lake1=[761.304,3368.038]
Lake2=[761.809,3367.584]
Lake3=[775.095,3338.919]
sites=[Lake1,Lake2,Lake3]
sites=np.array(sites)
#TEST GEOMETRY
# T1=np.array([[0,0],[1,1],[2,2]])
# T2=np.array([[1,1],[1,2],[1,3]])
# T3=np.array([[1,0],[2,0]])
# Faults=[T1,T2,T3]
#sites=np.array([[0,1],[2,1]])

#%% INPUT Spectific scenarios to test
#Segments of Xianshuihe Fault that ruptured for historical earthquakes

EQ2014=F2[8:14]
EQ1981=F1[24:38]
EQ1955=F3
EQ1973=np.concatenate([np.array([[609, 3515]]),F1[0:20]])
EQ1923=F1[11:30]
EQ1904=F1[15:33]
EQ1893=F1[32:54]
EQ1793=F1[41:50]
EQ1792=F1[23:32]
EQ1786=F2[22:43]
EQ1748=F2[0:16]
EQ1747=F1[16:26]
EQ1725=F2[11:30]
EQ1700=F1[53:67]
#%% INPUT Observations
obs=[True,False,True] #Was an earthquake recorded in the lake ?
sens=[5.5,6.5,6.5] #What is the lake intensity threshold for recording an Earthquake ?

#%% CODE Function definition 

#Divide the fault in all possible segments
def make_seg(Fault):
    seg_list = []
    for i in range(len(Fault)):
        for j in range(i+2, len(Fault)+1):
            seg_list.append(Fault[i:j])
    return seg_list


#Compute magnitude and intensity for a rupture
#You can change intensitiy prediction equation and magnitude-rupture length scaling law
def intensity(rupt, Lake):
    min_distance = float('inf')  # Start with a very high value
    px, py = Lake
    tot_len = 0
    
    for i in range(len(rupt) - 1):
        # Explore all segments composed of successive points
        ax, ay = rupt[i]
        bx, by = rupt[i + 1]
        AB = ((bx - ax)**2 + (by - ay)**2)
        AP_BP = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / AB
        
        if AP_BP < 0:
            distance = np.sqrt((px - ax)**2 + (py - ay)**2)
        elif AP_BP > 1:
            distance = np.sqrt((px - bx)**2 + (py - by)**2)
        else:
            qx, qy = ax + AP_BP * (bx - ax), ay + AP_BP * (by - ay)
            distance = np.sqrt((px - qx)**2 + (py - qy)**2)
        
        min_distance = min(min_distance, distance)
        tot_len += np.sqrt(AB)
        
    #Choose a scaling law for magnitude
    #Mag = 5.08 + 1.16 * np.log10(tot_len)  # Compute magnitude from scaling law of Wells & Coppersmith 1994
    #Mag = (2.943+np.log10(tot_len))/0.681 #Thingbaijam et al. 2017
    Mag=2/3*(2*(np.log10(tot_len)+8.11)-9.1) #Yen & Ma 2011
    
    #Select an Intensity prediction equation
    h=8 #epicenter depth for Bindi et al. 2011
    Intensity = 0.788 * Mag + 1.764 - 1.898 * np.log10(np.sqrt((min_distance**2+h**2)/h**2)) - 2.673e-3 * (np.sqrt(min_distance**2+h**2) - h) #Intensity prediction equation for central Asia, Bindi et al. 2011
    #Intensity=-3.6111*np.log10(min_distance+8)+4+Mag  #Custom
    #Intensity=4.448+1.0723*Mag-3.2255*np.log10(min_distance+9) #Liu et al. 2023
    #Intensity =3.950+0.913*Mag-1.107*np.log(np.sqrt(min_distance**2+(1+0.813*np.exp(Mag-5))**2)) #Intensity prediction equation for active crustal region Allen et al. 2012
    return Intensity, Mag, min_distance, tot_len

#Compute probability of a rupture to be registered by a Lake
#Adjust dispersion for the IPE used
def p(site,rupt,sensitivity):
    #probabiltity that rupture generates local intensity above the site snsitivity
    Int,Mag, min_distance,tot_len=intensity(rupt,site)
    sigma = 0.7 #standard deviation of IPE from Bindi et al. 2011 #error on Magnitude is neglictible
    #sigma = 0.82 #0.8 is the standard deviation of IPE from  Allen et al. 2012
    #(because I~Mag+cst and Mag as a variance of 0.2² the final variance is sqrt(0.2²+0.8²)~0.82)
    # probability of positive evidence P(X > sens)
    proba = 1-norm.cdf(sensitivity, Int, sigma)
    return proba, Mag

#Combines negatives and positives evidence to evaluate the rupture Probability
def prob(rupt, sites, obs, sens):
#If the rupture happen what is the Probability of having all the observations (negative obs[i]=0, or positive obs[i]=1) in the sites with different sensitivies (sens[i]).
    proba=1
    for i in range(len(obs)):
        if obs[i]:
            pr,M=p(sites[i],rupt,sens[i])
            proba=proba*pr
        else:
            pr,M=p(sites[i],rupt,sens[i]+0.5)
            proba=proba*(1-pr)
    norm_proba=proba**(1/len(obs)) #account for number of observations
    return proba,M

#Explore all possible ruptures to find the most likely scenario given observations
def search_rupture(Faults, sites, obs, sens):
    max_prob=0
    Probability=[]
    Mag_list=[]
    seg_all=[]
    for i in range(len(Faults)):
        Fault=Faults[i]
        Probability.append([])
        Mag_list.append([])
        seg_all.append([])
        seg_list=make_seg(Fault)
        seg_all[i].append(seg_list)
        for j in range(len(seg_list)):
            proba,M=prob(seg_list[j],sites,obs,sens)
            Probability[i].append(proba)
            Mag_list[i].append(M)
            if proba==max(max_prob,proba):
                max_prob=proba
                best_rupt=seg_list[j]
    return seg_all, Probability, Mag_list, best_rupt, max_prob

#%% OUTPUT Compute and plot intensities for a specific rupture scenario
Rupt=EQ1700 #Input the rupture geometry you want to test here
P_rupt,M=prob(Rupt, sites, obs, sens)

plt.figure(figsize=(5,7))

# Set grid dimensions
points_np = np.array([point for sub_list in Faults for point in sub_list])
min_x, min_y = points_np.min(axis=0)
max_x, max_y = points_np.max(axis=0)

#compute and plot intensity field for best scenario
x = np.linspace(min_x, max_x, num=100)  
y = np.linspace(min_y, max_y, num=100)  
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)

# Compute intensity for every grid point
for i in range(len(x)):
    for j in range(len(y)):
        point=np.array([x[i],y[j]])
        a,b,c,d = intensity(Rupt, point)
        Z[j,i]=a
contourf_plot = plt.contourf(X,Y, Z, levels=100, cmap='jet',vmin=3,vmax=8.5)
contour_lines = plt.contour(X,Y, Z, levels=[4,5,6,7,8,9,10], colors='w',linestyles='dashed')
cbar = plt.colorbar(contourf_plot,)
plt.clabel(contour_lines, fmt='%2.1f', colors='w', fontsize=14)
cbar.set_label('Intensity')

#plot lakes
for j in range(0,len(sites)):
    sx,sy=sites[j]
    o=obs[j]
    i=intensity(Rupt, sites[j])[0]
    plt.plot(sx, sy, 'o', color=(o/2,0,o), label='Lakes')
    plt.text(sx, sy,str(round(i,1)), fontsize=14, color=(o/2,0,o)) 
    
#plot fault map
for seg in Faults:
    Fx = seg[:, 0]
    Fy = seg[:, 1]
    plt.plot(Fx, Fy, color='black')

#plot rupture location
Bx = Rupt[:, 0]
By = Rupt[:, 1]
plt.plot(Bx, By, '-',linewidth=3, color='violet', label='Fault')


plt.axis('equal')
plt.title('rupture scenario\nM='+str(round(M,1))+'\nP='+str(round(P_rupt,3)))
plt.xlabel('Easting (km)')
plt.ylabel('Northing (km)')
plt.show()

#%% OUTPUT Find the best fitting rupture scenario

plt.figure(figsize=(5,6))
Sx = sites[:,0]
Sy = sites[:,1]

#look for best rupture scenario
seg_all, Probability,M,best_rupt, max_prob = search_rupture(Faults, sites, obs, sens)
Intensity, Mag, min_distance, tot_len=intensity(best_rupt,Lake1)

# Set grid dimensions
points_np = np.array([point for sub_list in [F1, F2, F3] for point in sub_list])
min_x, min_y = points_np.min(axis=0)
max_x, max_y = points_np.max(axis=0)

#compute and plot intensity field for best scenario
x = np.linspace(min_x, max_x, num=100)  
y = np.linspace(min_y, max_y, num=100)  
X, Y = np.meshgrid(x, y)


Z = np.zeros(X.shape)
for i in range(len(x)):
    for j in range(len(y)):
        point=np.array([x[i],y[j]])
        a,b,c,d = intensity(best_rupt, point)
        Z[j,i]=a
contourf_plot = plt.contourf(X,Y, Z, levels=100, cmap='jet',vmin=3,vmax=8.5)
contour_lines = plt.contour(X,Y, Z, levels=[4,5,6,7,8,9,10], colors='w',linestyles='dashed')
cbar = plt.colorbar(contourf_plot,)
plt.clabel(contour_lines, fmt='%2.1f', colors='w', fontsize=14)
cbar.set_label('Intensity')

#plot lakes and fault
for j in range(0,len(sites)):
    sx,sy=sites[j]
    o=obs[j]
    i=intensity(best_rupt, sites[j])[0]
    plt.plot(sx, sy, 'o', color=(o/2,0,o), label='Lakes')
    plt.text(sx, sy,str(round(i,1)), fontsize=14, color=(o/2,0,o)) 
    
for Fault in [F1, F2, F3]:
        Fx = Fault[:, 0]
        Fy = Fault[:, 1]
        plt.plot(Fx, Fy, color='k')

#plot best rupture location
Bx = best_rupt[:, 0]
By = best_rupt[:, 1]
plt.plot(Bx, By, '-',linewidth=3, color='violet', label='Fault')
plt.axis('equal')
plt.title('Best rupture scenario\nM='+str(round(Mag,1))+'\nP='+str(round(max_prob,3)))
plt.xlabel('Easting (km)')
plt.ylabel('Northing (km)')
plt.show()