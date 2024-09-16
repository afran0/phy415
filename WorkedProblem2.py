#!/usr/bin/env python
# coding: utf-8

# In[53]:


from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

k1 = 0.5  # cooling constant for object
k_env = 0.2  # cooling constant for env temperature
T_target = 50  # target temperature

def system(t, y):
    T_obj, T_env = y
    dT_obj_dt = -k1 * (T_obj - T_env)  #cooling of object 1
    dT_env_dt = -k_env * (T_env - T_target)  #enviornment temperature
    return [dT_obj_dt, dT_env_dt]

T_obj_0 = 150  # initial temperature of the object
T_env_0 = 70  # initial ambient temperature
y0 = [T_obj_0, T_env_0]

t_span = (0, 70)
t_eval = np.linspace(t_span[0], t_span[1], 500)

sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

plt.plot(sol.t, sol.y[0], label='Object Temperature')
plt.plot(sol.t, sol.y[1], label='Ambient Temperature', linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title('Cooling of Object with Variable Ambient Temperature')
plt.show()

plt.subplot(1, 2, 2)
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('Object Temperature')
plt.ylabel('Ambient Temperature')
plt.title('Phase Space Diagram')


# In[52]:


from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

k1 = 0.5  # cooling constant for object
A = 1  # amplitude of temperature oscillation
omega = 0.1  # frequency of oscillation

def system(t, y):
    T_obj, T_env = y
    dT_obj_dt = -k1 * (T_obj - T_env)  # cooling of object 1
    dT_env_dt = A * np.sin(omega * t)  # oscillating ambient temperature
    return [dT_obj_dt, dT_env_dt]

T_obj_0 = 150  # initial temperature of the object
T_env_0 = 70  # initial ambient temperature
y0 = [T_obj_0, T_env_0]

t_span = (0, 80)
t_eval = np.linspace(t_span[0], t_span[1], 500)

sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

plt.plot(sol.t, sol.y[0], label='Object Temperature')
plt.plot(sol.t, sol.y[1], label='Ambient Temperature', linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title('Object Cooling with Oscillating Ambient Temperature')
plt.show()

plt.subplot(1, 2, 2)
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('Object Temperature')
plt.ylabel('Ambient Temperature')
plt.title('Phase Space Diagram')

plt.tight_layout()
plt.show()


# In[50]:


from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

k1 = 0.5  # cooling constant for object 1
k2 = 0.4  # cooling constant for object 2
alpha = 0.1  # heat exchange rate between object 1 and object 2
A = 5 # amplitude of ambient temperature oscillation
omega = 0.8  # frequency of oscillation

def system(t, y):
    T1, T2, T_env = y
    dT1_dt = -k1 * (T1 - T_env) - alpha * (T1 - T2)
    dT2_dt = -k2 * (T2 - T_env) + alpha * (T1 - T2)
    dT_env_dt = (A * np.sin(omega * t)) # ADD tempertaure factors due to object 1 &2 
    return [dT1_dt, dT2_dt, dT_env_dt]

t_span = (0, 10) 
y0 = [50,80,70]  # temperatures of T1, T2, and T_env

sol = solve_ivp(system, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 500))

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Temperature of Object 1', color='purple')
plt.plot(sol.t, sol.y[1], label='Temperature of Object 2', color='green')
plt.plot(sol.t, sol.y[2], label='Ambient Temperature', color='orange', linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.title('Temperature Evolution Over Time')
plt.xlim(t_span)
plt.ylim(bottom=None, top=90)
plt.show()


# In[ ]:





# In[ ]:




