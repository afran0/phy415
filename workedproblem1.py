import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#parameters
g = 9.81
m1, m2 = 1.0, 1.0
l1, l2 = 1.0, 1.0
k = 5.0

#equations of motion
def coupled_pendulum(t, y):
    theta1, omega1, theta2, omega2 = y
    
    theta1_dt = omega1
    omega1_dt = -(g/l1) * theta1 - (k/m1) * (theta1 - theta2)
    
    theta2_dt = omega2
    omega2_dt = -(g/l2) * theta2 + (k/m2) * (theta1 - theta2)
    return [theta1_dt, omega1_dt, theta2_dt, omega2_dt]

initial_conditions_1 = [0.2, 0.0, 0.2, 0.0] 
initial_conditions_2 = [0.1, 0.0, -0.1, 0.0]

t_span = (0, 10)  #time interval
t_eval = np.linspace(t_span[0], t_span[1], 1000)  #time points

#numerical solutions
sol_1 = solve_ivp(coupled_pendulum, t_span, initial_conditions_1, t_eval=t_eval)
sol_2 = solve_ivp(coupled_pendulum, t_span, initial_conditions_2, t_eval=t_eval)

plt.figure(figsize=(10, 6))

#subplot 1
plt.subplot(2, 1, 1)
plt.plot(sol_1.t, sol_1.y[0], label=r'$\theta_1$', color='b')
plt.plot(sol_1.t, sol_1.y[2], label=r'$\theta_2$', linestyle='dashed', color='r')
plt.title('Theta Values-Symmetric Displacement With No Initial Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()

#subplot 2
plt.subplot(2, 1, 2)
plt.plot(sol_2.t, sol_2.y[0], label=r'$\theta_1$', color='b')
plt.plot(sol_2.t, sol_2.y[2], label=r'$\theta_2$', linestyle='dashed', color='r')
plt.title('Theta Values-Asymmetric Displacement With No Initial Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()

plt.tight_layout()
plt.show()

initial_conditions_3 = [0.1, 0.5, -0.1, -0.5]  #displacement with initial velocities
initial_conditions_4 = [0.2, -0.5, 0.1, 0.5]  #displacement with opposite initial velocities

sol_3 = solve_ivp(coupled_pendulum, t_span, initial_conditions_3, t_eval=t_eval)
sol_4 = solve_ivp(coupled_pendulum, t_span, initial_conditions_4, t_eval=t_eval)

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(sol_3.t, sol_3.y[0], label=r'$\theta_1$', color='b')
plt.plot(sol_3.t, sol_3.y[2], label=r'$\theta_2$', linestyle='dashed', color='r')
plt.title('Theta Values-Symmetric Displacement With Initial Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sol_4.t, sol_4.y[0], label=r'$\theta_1$', color='b')
plt.plot(sol_4.t, sol_4.y[2], label=r'$\theta_2$', linestyle='dashed', color='r')
plt.title('Theta Values-Asymmetric Displacement With Initial Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()

plt.tight_layout()
plt.show()
