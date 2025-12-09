#!/usr/bin/env python
# coding: utf-8

# # Dichiaro Moduli utili e definizioni interessanti

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# In[2]:


def trova_massimi(time, voltage, prominence):
    # Trova i massimi locali con una soglia di prominenza per evitare falsi massimi
    peaks, _ = find_peaks(voltage, prominence=prominence)  # Soglia regolabile
    time_peaks = time[peaks]
    voltage_peaks = voltage[peaks]
    return time_peaks, voltage_peaks


# In[3]:


def trova_minimi(time, voltage, prominence):
    # Trova i minimi locali con una soglia di prominenza per evitare falsi minimi
    troughs, _ = find_peaks(-voltage, prominence=prominence)  # Soglia regolabile
    time_troughs = time[troughs]
    voltage_troughs = voltage[troughs]
    return time_troughs, voltage_troughs


# In[4]:


def Errore_R(R):

    return R*0.1/100 + 4



# In[5]:


def Errore_C(C):

    return C*0.1


# In[6]:


def Errore_tau(Tau, incertezza_C, incertezza_R):

    incertezza_Tau = np.sqrt(incertezza_C**2 + incertezza_R**2)
    return Tau*incertezza_Tau 


# In[7]:


def Tau_with_c(x, A, tau, C):

    return (A * np.exp(-x/tau) + C) 


# In[8]:


def De_Tau_with_c(x, A, tau, C):

    return (-A/tau) * np.exp(-x/tau)


# In[9]:


def R_Parassita(Tau_r, C, R):

    return (Tau_r/C) - R


# In[10]:


def Errore_R_Parassita(incertezza_Tau_r, incertezza_C, Err_R, C , Tau_r):

    return np.sqrt((np.sqrt(incertezza_Tau_r**2 + incertezza_C**2)*(Tau_r / C))**2 + Err_R**2)


# In[11]:


def Cof_R(x ,A, w0 , C ):

    return (A/(1 - x * w0 )) + C


# # Circuito RC con R pari a 573Ohm e C pari a 2.2 e-3 F

# In[12]:


file_name = "Dente%2%R573.xlsx"
df = pd.read_excel(file_name)
#print(df)
Col_1 = "Time"
Col_2 = "Vc"
Col_3 = "Vg"
t = df[Col_1].to_numpy()
Vc = df[Col_2].to_numpy()
Vg = df[Col_3].to_numpy()


# In[13]:


C = 2.2e-3
R = 573
tau_i = C*R 

#Calcolo Errori di C R e Tau

Err_R = Errore_R(R)
Err_C = Errore_C(C)
inc_R = Err_R/R
inc_C = Err_C/C
Err_Tau_i = Errore_tau(tau_i, inc_C, inc_R)

i = 0
k = 0

Maxs, _ = find_peaks(Vc, prominence= 1)
Mins, _ = find_peaks(-Vc, prominence= 1)
dt_1 = t[Mins[i]:Maxs[k]:1]
dvc_1 = Vc[Mins[i]:Maxs[k]:1]
Error_Vg = 0.001

delta = dt_1 - t[Mins[i]]


# In[14]:


popt, pcov = curve_fit(Tau_with_c, delta, dvc_1, p0 = [1, (1/tau_i),-1],sigma = Error_Vg, absolute_sigma = True)

A_fit, Tau_fit, C_fit = popt


Disc_Tau = pcov[1,1]
print(f"Il Tau reale vale: {Tau_fit} +/- {np.sqrt(Disc_Tau)}")
print(f"\nIl Tau ideale vale: {tau_i} +/- {Err_Tau_i}")

y_fit = Tau_with_c(delta, A_fit, Tau_fit, C_fit)
y_dati = dvc_1

sigma_x = 0.005**2
sigma = y_fit - y_dati
err_y = np.sqrt(De_Tau_with_c(delta, A_fit, Tau_fit, C_fit)**2 + sigma_x)

y_0 = np.zeros(len(delta))
taus = Tau_fit, tau_i
err_Taus = np.sqrt(Disc_Tau) * 3, Err_Tau_i
index = ["Tau_R","Tau_I"]


# In[15]:


plt.figure(figsize = (10,8))

plt.subplot(2, 2, 1)
plt.plot(delta, dvc_1, color = 'b', label = "Dati")
plt.plot(delta, y_fit, color= 'r', label= "Fit esponenziale")
plt.grid(True)
plt.legend()
plt.subplot(2,2,3)
plt.errorbar(delta, sigma, yerr = err_y, fmt = '.' , color = 'g', label = "Scarti", capsize = 3)
#plt.plot(delta, sigma, color = 'y', label = "scarti")
plt.plot(delta, y_0, color = 'r', label = "y = 0")
plt.ylim(-0.01,0.02)
plt.xlim(0 , 0.1)
plt.legend()
plt.grid(True)
plt.subplot(2, 2 , 2)
plt.errorbar(index, taus, yerr = err_Taus, fmt ='o', color ='r', capsize = 3)
plt.ylim(1,1.6)
plt.xlim(-1 , 2)

plt.grid(True)


# # Circuito RC con R pari a 215 Ohm e C pari a 2.2 e-3 F

# In[16]:


file_name = "Dente 1 R215.xlsx"

df = pd.read_excel(file_name)
#print(df)
Col_1 = "Tempi"
Col_2 = "Vc"
Col_3 = "Vg"
t_1 = df[Col_1].to_numpy()
Vc_1 = df[Col_2].to_numpy()
Vg_1 = df[Col_3].to_numpy()


# In[17]:


C = 2.2e-3
R = 215
tau_i = C*R 

Err_R = Errore_R(R)
Err_C = Errore_C(C)
inc_R = Err_R/R
inc_C = Err_C/C
Err_Tau_i = Errore_tau(tau_i, inc_C, inc_R)


i = 44
k = 1


Maxs, _ = find_peaks(Vc_1, prominence = 1)
Mins, _ = find_peaks(-Vc_1, prominence = 1)



dt_2 = t_1[Mins[i]:Maxs[k]:1]
dvc_2 = Vc_1[Mins[i]:Maxs[k]:1]
Error_Vg = 0.001

delta_1 = dt_2 - t_1[Mins[i]]


# In[18]:


popt, pcov = curve_fit(Tau_with_c, delta_1, dvc_2, p0 = [2, (1/tau_i),-1],sigma = Error_Vg, absolute_sigma = True)

A_fit, Tau_fit, C_fit = popt
print(f"\nTau Ideale: {tau_i} +/- {Err_Tau_i}")
Disc_Tau = pcov[1,1]
Err_Tau_r = np.sqrt(Disc_Tau) * 3
print(f"Tau Reale: {Tau_fit} +/- {Err_Tau_r}")

taus = Tau_fit, tau_i
err_Taus = Err_Tau_r, Err_Tau_i
#err_A = np.sqrt(pcov[0,0])
index = ["Tau_R","Tau_I"]
#print(f"Ampiezza: {A_fit} +/- {err_A} ")


# In[19]:


plt.figure(figsize = (10,8))

plt.subplot(2, 2, 1)
plt.plot(delta_1, dvc_2)
#plt.ylim(1.01, 1.03)
plt.grid(True)
plt.subplot(2, 2 , 2)
plt.errorbar(index, taus, yerr = err_Taus, fmt ='o', color ='r', capsize = 3)
plt.grid(True)


# # Circuito RC con R pari a 1397 Ohm e C pari a 2.2 e-3 F

# In[20]:


file_name = "Dente 3 R1397.xlsx"

df = pd.read_excel(file_name)
#print(df)
Col_1 = "Tempi"
Col_2 = "Vc"
Col_3 = "Vg"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[21]:


C = 2.2e-3
R = 1397
tau_i = C*R 

Maxs, _ = find_peaks(Vc_2, prominence = 0.01)
Mins, _ = find_peaks(-Vc_2, prominence = 0.01)

Err_R = Errore_R(R)
Err_C = Errore_C(C)
inc_R = Err_R/R
inc_C = Err_C/C
Err_Tau_i = Errore_tau(tau_i, inc_C, inc_R)


i = 13335
k = 17000


dt_3 = t_2[i:k:1]
dvc_3 = Vc_2[i:k:1]
Error_Vg = 0.001

#print(f"Stampo i Massimi: {Maxs}\n E i Minimi: {Mins}")

delta_3 = dt_3 - t_2[i]


# In[22]:


popt, pcov = curve_fit(Tau_with_c, delta_3, dvc_3, p0 = [1, (1/tau_i),-1],sigma = Error_Vg, absolute_sigma = True)

A_fit, Tau_fit, C_fit = popt
print(f"\nTau Ideale: {tau_i} +/- {Err_Tau_i}")
Disc_Tau = pcov[1,1]
Err_Tau_r = np.sqrt(Disc_Tau)
print(f"Tau Reale: {Tau_fit} +/- {Err_Tau_r}")

taus = Tau_fit, tau_i
err_Taus = Err_Tau_r, Err_Tau_i
err_A = np.sqrt(pcov[0,0])
index = ["Tau_R","Tau_I"]

#print(f"Ampiezza: {A_fit} +/- {err_A} ")


# In[23]:


plt.figure(figsize = (10,8))

plt.subplot(2,2,1)
plt.plot(delta_3,dvc_3)
plt.grid(True)
plt.subplot(2, 2 , 2)
plt.errorbar(index, taus, yerr = err_Taus, fmt ='o', color ='r', capsize = 3)
plt.grid(True)


# # Circuito RC Con Resistenza R215 Ohm e C pari a 1000 miF

# In[24]:


file_name = "Dente 1 C1000 miF.xlsx"

df = pd.read_excel(file_name)
#print(df)
Col_1 = "Tempi"
Col_2 = "Vc"
Col_3 = "Vg"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[25]:


C = 1e-3
R = 215
tau_i = C*R 

Maxs, _ = find_peaks(Vc_2, prominence = 0.01)
Mins, _ = find_peaks(-Vc_2, prominence = 0.01)

Err_R = Errore_R(R)
Err_C = Errore_C(C)
inc_R = Err_R/R
inc_C = Err_C/C
Err_Tau_i = Errore_tau(tau_i, inc_C, inc_R)

i = 9
k = 1


dt_3 = t_2[Mins[i]:Maxs[k]:1]
dvc_3 = Vc_2[Mins[i]:Maxs[k]:1]
Error_Vg = 0.001

#print(f"Stampo i Massimi: {Maxs}\n E i Minimi: {Mins}")

delta_3 = dt_3 - t_2[i]


# In[26]:


popt, pcov = curve_fit(Tau_with_c, delta_3, dvc_3, p0 = [1, (1/tau_i),-1],sigma = Error_Vg, absolute_sigma = True)

A_fit, Tau_fit, C_fit = popt
print(f"\nTau Ideale: {tau_i} +/- {Err_Tau_i}")
Disc_Tau = pcov[1,1]
Err_Tau_r = np.sqrt(Disc_Tau) * 3
print(f"Tau Reale: {Tau_fit} +/- {Err_Tau_r}")

taus = Tau_fit, tau_i
err_Taus = Err_Tau_r, Err_Tau_i
index = ["Tau_R","Tau_I"]


# In[27]:


plt.figure(figsize = (10,8))

plt.subplot(2,2,1)
plt.plot(delta_3,dvc_3 + 1)
plt.grid(True)
plt.subplot(2, 2 , 2)
plt.errorbar(index, taus, yerr = err_Taus, fmt ='o', color ='r', capsize = 3)
plt.grid(True)


# # Circuito RC Con Resistenza R215 Ohm e C pari a 470 miF

# In[28]:


file_name = "Dente 2 C470 miF.xlsx"

df = pd.read_excel(file_name)
#print(df)
Col_1 = "Tempi"
Col_2 = "Vc"
Col_3 = "Vg"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[29]:


C = 470e-6
R = 215
tau_i = C*R 

Maxs, _ = find_peaks(Vc_2, prominence = 0.01)
Mins, _ = find_peaks(-Vc_2, prominence = 0.01)

Err_R = Errore_R(R)
Err_C = Errore_C(C)
inc_R = Err_R/R
inc_C = Err_C/C
Err_Tau_i = Errore_tau(tau_i, inc_C, inc_R)

i = len(Mins) - 1
k = len(Maxs) - 1


dt_3 = t_2[Mins[i]:Maxs[k]:1]
dvc_3 = Vc_2[Mins[i]:Maxs[k]:1]
Error_Vg = 0.001

#print(f"Stampo i Massimi: {Maxs}\n E i Minimi: {Mins}")

delta_3 = dt_3 - t_2[Mins[i]]


# In[30]:


popt, pcov = curve_fit(Tau_with_c, delta_3, dvc_3, p0 = [1, (1/tau_i),-1],sigma = Error_Vg, absolute_sigma = True)

A_fit, Tau_fit, C_fit = popt
print(f"\nTau Ideale: {tau_i} +/- {Err_Tau_i}")
Disc_Tau = pcov[1,1]
Err_Tau_r = np.sqrt(Disc_Tau) * 3
print(f"Tau Reale: {Tau_fit} +/- {Err_Tau_r}")

taus = Tau_fit, tau_i
err_Taus = Err_Tau_r, Err_Tau_i
index = ["Tau_R","Tau_I"]


# In[31]:


plt.figure(figsize = (10,8))

plt.subplot(2,2,1)
plt.plot(delta_3,dvc_3 + 1)
plt.grid(True)
plt.subplot(2, 2 , 2)
plt.errorbar(index, taus, yerr = err_Taus, fmt ='o', color ='r', capsize = 3)
plt.grid(True)


# # Funzione Av al variare della frequenza wo pari a 0.5 hz

# In[32]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "0.5"
Col_1 = f"Time " + Frequenza
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[33]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_0_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_0_5)
Err_Av = np.std(Av_0_5,ddof = 1)/np.sqrt(len(Av_0_5) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests = []
Av_Bests.append(Av_best)

Errs_Av = []
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 1 hz

# In[34]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "1.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[35]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_1 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_1)
Err_Av = np.std(Av_1,ddof = 1)/np.sqrt(len(Av_1) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 1.5 hz

# In[36]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "1.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[37]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_1_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_1_5)
Err_Av = np.std(Av_1_5,ddof = 1)/np.sqrt(len(Av_1_5) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 2 hz

# In[38]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "2.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[39]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_2 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_2)
Err_Av = np.std(Av_2,ddof = 1)/np.sqrt(len(Av_2) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 2.5 hz

# In[40]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "2.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[41]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_2_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_2_5)
Err_Av = np.std(Av_2_5,ddof = 1)/np.sqrt(len(Av_2_5) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 3 hz

# In[42]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "3.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[43]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_3 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_3)
Err_Av = np.std(Av_3,ddof = 1)/np.sqrt(len(Av_3) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 3.5 hz

# In[44]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "3.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[45]:


PPeaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
Vg_Peaks = Vg_Peaks[0:len(Vc_Peaks):1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_3_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_3_5)
Err_Av = np.std(Av_3_5,ddof = 1)/np.sqrt(len(Av_3_5) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 4hz

# In[46]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "4.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[47]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_4 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_4)
Err_Av = np.std(Av_4,ddof = 1)/np.sqrt(len(Av_4) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 4.5 hz

# In[48]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "4.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[49]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_4_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_4_5)
Err_Av = np.std(Av_4_5,ddof = 1)/np.sqrt(len(Av_4_5) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 5 hz

# In[50]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "5.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[51]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_5)
Err_Av = np.std(Av_5,ddof = 1)/np.sqrt(len(Av_5) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 5.5 hz

# In[52]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "5.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[53]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_5_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_5_5)
Err_Av = np.std(Av_5_5,ddof = 1)/np.sqrt(len(Av_5_5) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 6 hz

# In[54]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "6.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[55]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_6 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_6)
Err_Av = np.std(Av_6,ddof = 1)/np.sqrt(len(Av_6) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 6.5hz

# In[56]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "6.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[57]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_6_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_6_5)
Err_Av = np.std(Av_6_5,ddof = 1)/np.sqrt(len(Av_6_5) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 7hz

# In[58]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "7.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[59]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_7 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_7)
Err_Av = np.std(Av_7,ddof = 1)/np.sqrt(len(Av_7) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 7.5hz

# In[60]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "7.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[61]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_7_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_7_5)
Err_Av = np.std(Av_7_5,ddof = 1)/np.sqrt(len(Av_7_5) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 8hz

# In[62]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "8.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[63]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_8 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_8)
Err_Av = np.std(Av_8,ddof = 1)/np.sqrt(len(Av_8) -  1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 8.5hz

# In[64]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "8.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[65]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_8_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_8_5)
Err_Av = np.std(Av_8_5,ddof = 1)/np.sqrt(len(Av_8_5) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 9hz

# In[66]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "9.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[67]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_9 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_9)
Err_Av = np.std(Av_9,ddof = 1)/np.sqrt(len(Av_9) - 1)

print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 9.5hz

# In[68]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "9.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[69]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_9_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_9_5)
Err_Av = np.std(Av_9_5,ddof = 1)/np.sqrt(len(Av_9_5) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 10hz

# In[70]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "10.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[71]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_10 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_10)
Err_Av = np.std(Av_10 ,ddof = 1)/np.sqrt(len(Av_10) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 10.5hz

# In[72]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "10.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[73]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_10_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_10_5)
Err_Av = np.std(Av_10_5 ,ddof = 1)/np.sqrt(len(Av_10_5) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 11hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "11.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_11 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_11)
Err_Av = np.std(Av_11 ,ddof = 1)/np.sqrt(len(Av_11) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 11.5hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "11.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_11_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_11_5)
Err_Av = np.std(Av_11_5 ,ddof = 1)/np.sqrt(len(Av_11_5) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 12hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "12.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_12 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_12)
Err_Av = np.std(Av_12 ,ddof = 1)/np.sqrt(len(Av_12) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 12.5hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "12.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_12_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_12_5)
Err_Av = np.std(Av_12_5 ,ddof = 1)/np.sqrt(len(Av_12_5) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 13hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "13.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_13 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_13)
Err_Av = np.std(Av_13 ,ddof = 1)/np.sqrt(len(Av_13) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 13.5hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "13.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_13_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_13_5)
Err_Av = np.std(Av_13_5 ,ddof = 1)/np.sqrt(len(Av_13_5) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 14hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "14.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_14 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_14)
Err_Av = np.std(Av_14 ,ddof = 1)/np.sqrt(len(Av_14) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 14.5hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "14.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_14_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_14_5)
Err_Av = np.std(Av_14_5 ,ddof = 1)/np.sqrt(len(Av_14_5) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 15hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "15.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_15 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_15)
Err_Av = np.std(Av_15 ,ddof = 1)/np.sqrt(len(Av_15) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 15.5hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "15.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_15_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_15_5)
Err_Av = np.std(Av_15_5 ,ddof = 1)/np.sqrt(len(Av_15_5) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 16hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "16.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_16 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_16)
Err_Av = np.std(Av_16 ,ddof = 1)/np.sqrt(len(Av_16) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 16.5hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "16.5"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

#Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_16_5 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_16_5)
Err_Av = np.std(Av_16_5 ,ddof = 1)/np.sqrt(len(Av_16_5) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Funzione Av al variare della frequenza wo pari a 17hz

# In[ ]:


file_name = "Frequenze.xlsx"

df = pd.read_excel(file_name)
#print(df)
Frequenza = "17.0"
Col_1 = f"Time {Frequenza}"
Col_2 = f"Vc {Frequenza}hz"
Col_3 = f"Vg {Frequenza}hz"
t_2 = df[Col_1].to_numpy()
Vc_2 = df[Col_2].to_numpy()
Vg_2 = df[Col_3].to_numpy()


# In[ ]:


Peaks_Vc = trova_massimi(t_2, Vc_2, 0.01)
Throughts_Vc = trova_minimi(t_2, Vc_2, 0.01)

Peaks_Vg = trova_massimi(t_2, Vg_2, 0.01)
Throughts_Vg = trova_minimi(t_2, Vg_2, 0.01)


Vg_Peaks = Peaks_Vg[1]
Vc_Peaks = Peaks_Vc[1]

#Vc_Peaks = Vc_Peaks[:-1]
#Vg_Peaks = Vg_Peaks[:-1]

Vg_Throughts = Throughts_Vg[1]
Vc_Throughts = Throughts_Vc[1]

Vg_Throughts = Vg_Throughts[:-1]
#Vc_Throughts = Vc_Throughts[:-1]

Peaks_Av = np.abs(Vc_Peaks/Vg_Peaks)
Throughts_Av = np.abs(Vc_Throughts/Vg_Throughts)

Av_17 = np.concatenate((Peaks_Av, Throughts_Av))
Av_best = np.mean(Av_17)
Err_Av = np.std(Av_17 ,ddof = 1)/np.sqrt(len(Av_17) - 1)
print(f"Av best: {Av_best} +/- {Err_Av}")
Av_Bests.append(Av_best)
Errs_Av.append(Err_Av)


# # Grafico Av al variare di Wo:

# In[ ]:


Av_best = np.array(Av_Bests)
Err_Av = np.array(Errs_Av)

#print(Av_best)
w = []
for i in range(1,35):
    w.append(i/2)


wo = np.array(w)
#print(wo)


# # Stima di Cof Reale

# In[ ]:


Pi = np.pi
w0_i = 1/(tau_i * 2*Pi)
popt, pcov = curve_fit(Cof_R, wo, Av_best, p0 = [1 ,w0_i ,0],sigma = Err_Av, absolute_sigma = True)
A_Fit, w0_fit, C_fit = popt
Disc_wo = pcov[0,0]
Err_wo_r = np.sqrt(Disc_wo)*3
Err_wo_i = 0.001
print(f"La Cut-off Frequency Ideale: {w0_i} +/- {Err_wo_i}\nLa Cut-off Frequency Reale: {w0_fit} +/- {Err_wo_r}")
w0s = w0_i, w0_fit
Err_w0s = Err_wo_i, Err_wo_r
index = ["wo Reale","wo Ideale"]

F_w0 = Cof_R(wo ,A_fit, w0_fit, C_fit)
tau_r = 1/(w0_fit * 2*Pi)
print(tau_r)


# In[ ]:


plt.figure(figsize = (10,8))
plt.subplot(2,2,1)
plt.errorbar(wo, Av_best, yerr = Err_Av,xerr = 0.001, fmt ='.', color ='b', capsize = 3)
#plt.plot(wo, F_w0 , color = 'r', label= "Fit Funzione")
plt.grid(True)
plt.subplot(2,2,2)
plt.xlim(-1,2)
plt.ylim(0.12,0.14)
plt.errorbar(index, w0s, yerr = Err_w0s, fmt ='.', color ='r', capsize = 3)
plt.grid(True)


# # Stessi Calcoli Visti in precedenza nei circuiti Con C fisso e R variabile e vice versa ma considerando una R parassita di C

# In[ ]:


file_name = "Dente%2%R573.xlsx"
df = pd.read_excel(file_name)
#print(df)
Col_1 = "Time"
Col_2 = "Vc"
Col_3 = "Vg"
t = df[Col_1].to_numpy()
Vc = df[Col_2].to_numpy()
Vg = df[Col_3].to_numpy()


# In[ ]:


#Calcolo Errori di C R e Tau
C = 2.2e-3
R = 573
tau_i = R * C
Err_R = Errore_R(R)
Err_C = Errore_C(C)
inc_R = Err_R/R
inc_C = Err_C/C
Err_Tau_i = Errore_tau(tau_i, inc_C, inc_R)



i = 0
k = 0

Maxs, _ = find_peaks(Vc, prominence= 1)
Mins, _ = find_peaks(-Vc, prominence= 1)
dt_1 = t[Mins[i]:Maxs[k]:1]
dvc_1 = Vc[Mins[i]:Maxs[k]:1]
Error_Vg = 0.001

delta = dt_1 - t[Mins[i]]


# In[ ]:


popt, pcov = curve_fit(Tau_with_c, delta, dvc_1, p0 = [1, (1/tau_i),-1],sigma = Error_Vg, absolute_sigma = True)

A_fit, Tau_fit, C_fit = popt


Disc_Tau = pcov[1,1]
print(f"Il Tau reale vale: {Tau_fit} +/- {np.sqrt(Disc_Tau)}")
print(f"\nIl Tau ideale vale: {tau_i} +/- {Err_Tau_i}")

#y_fit = Tau_with_c(delta, A_fit, Tau_fit, C_fit)
#y_dati = dvc_1

#sigma_x = 0.005**2
#sigma = y_fit - y_dati
#err_y = np.sqrt(De_Tau_with_c(delta, A_fit, Tau_fit, C_fit)**2 + sigma_x)
Err_Tau_r =  np.sqrt(Disc_Tau)
#y_0 = np.zeros(len(delta))
taus = Tau_fit, tau_i
err_Taus = Err_Tau_r, Err_Tau_i
#index = ["Tau_R","Tau_I"]


Err_Tau_r = err_Taus[0]
incertezza_Tau_r = Err_Tau_r/Tau_fit

R_parassita = R_Parassita(Tau_fit, C, R)
Err_R_P = Errore_R_Parassita(incertezza_Tau_r, inc_C, Err_R, C,Tau_fit)

R_eq = R + R_parassita

inc_R_P = Err_R_P/R_parassita

tau_i_p = C * (R_eq)
Err_Tau_i_p = Errore_tau(tau_i_p, inc_C, inc_R_P)

print(f"La Resistenza parassita si ipotizza valga: {R_parassita} +/- {Err_R_P}")

print(f"Il tau considerando la resistenza parassita vale: {tau_i_p} +/- {Err_Tau_i_p}")


# In[ ]:


plt.figure(figsize = (10,8))
plt.xlim(2.5,120)
plt.ylim(-1.1,1.6)
plt.plot(t,Vc, color = 'r', label = "Funzione dente di sega")
plt.plot(t,Vg, color = 'y', label = "Onda Quadra")
plt.grid()
plt.legend(loc="upper left")


# In[ ]:




