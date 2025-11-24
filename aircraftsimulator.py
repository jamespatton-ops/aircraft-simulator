# A python based aircraft simulator
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
except ImportError:
    import sys
    sys.exit("Missing packages: numpy, scipy, matplotlib.\nActivate your venv and run:\n  python -m pip install numpy scipy matplotlib")


class AircraftSimulator:
    def __init__(self):
        # Aircraft parameters
        self.mass = 1000  # kg
        self.Ixx = 1200  # kg*m^2
        self.Iyy = 1800  # kg*m^2
        self.Izz = 2000  # kg*m^2
        self.Ixz = 0  # kg*m^2
        self.wing_area = 16.2  # m^2
        self.wing_span = 11.0  # m
        self.mean_chord = 1.5  # m

        # Flight condition
        self.U0 = 60.0  # m/s (cruise speed)
        self.rho = 1.225  # kg/m^3 (air density)
        self.theta0 = 0.0  # rad (trim pitch angle)

        # Stability derivatives
        self.set_stability_derivatives()

        # state variables [u, w, q, theta, v, p, r, phi]
        self.state = np.zeros(8)

        # Control inputs [elevator, aileron, rudder, throttle]
        self.controls = np.zeros(4)

        # Storage for plotting
        self.history = {'time': [], 'state': [], 'controls': []}

    def set_stability_derivatives(self):
        """Set stability derivatives for our aircraft"""
        # Longitudinal derivatives
        self.Xu = -0.05
        self.Xw = 0.1
        self.Zu = -0.3
        self.Zw = -2.0
        self.Zw_dot = -0.5
        self.Zq = -1.0
        self.Mu = 0.0
        self.Mw = -0.8
        self.Mw_dot = -1.0
        self.Mq = -2.0

        # Lateral derivatives
        self.Yv = -0.4
        self.Yp = 0.0
        self.Yr = 0.0
        self.Lv = -0.2
        self.Lp = -0.8
        self.Lr = 0.2
        self.Nv = 0.1
        self.Np = -0.1
        self.Nr = -0.2

        # Control derivatives
        self.Zde = -0.3
        self.Mde = -1.0
        self.Yda = 0.1
        self.Lda = 0.1
        self.Nda = -0.01
        self.Ydr = 0.1
        self.Ldr = 0.01
        self.Ndr = -0.1

    def set_controls(self, elevator=0, aileron=0, rudder=0, throttle=0):
        """Set control surface deflections (radians)"""
        self.controls = np.array([
            np.clip(elevator, -0.3, 0.3),
            np.clip(aileron, -0.3, 0.3),
            np.clip(rudder, -0.4, 0.4),
            np.clip(throttle, 0, 1)
        ])

    def longitudinal_dynamics(self, t, state_long):
        """Longitudinal equations of motion (u, w, q, theta)"""
        u, w, q, theta = state_long
        de = self.controls[0]

        # Force equations
        u_dot = (self.Xu * u + self.Xw * w) / self.mass - 9.81 * theta
        w_dot = (self.Zu * u + self.Zw * w + self.Zq * q + self.Zde * de) / self.mass + self.U0 * q

        # Moment equation (pitch)
        q_dot = (self.Mu * u + self.Mw * w + self.Mq * q + self.Mde * de) / self.Iyy

        # Kinematic
        theta_dot = q

        return [u_dot, w_dot, q_dot, theta_dot]

    def lateral_dynamics(self, t, state_lat):
        """Lateral-directional equations of motion (v, p, r, phi)"""
        v, p, r, phi = state_lat
        da, dr = self.controls[1], self.controls[2]

        # Force equations
        v_dot = (self.Yv * v + self.Yp * p + self.Yr * r +
                 self.Yda * da + self.Ydr * dr) / self.mass - self.U0 * r + 9.81 * phi

        # Moment equations
        p_dot = (self.Lv * v + self.Lp * p + self.Lr * r +
                 self.Lda * da + self.Ldr * dr) / self.Ixx
        r_dot = (self.Nv * v + self.Np * p + self.Nr * r +
                 self.Nda * da + self.Ndr * dr) / self.Izz

        # Kinematic
        phi_dot = p

        return [v_dot, p_dot, r_dot, phi_dot]

    # Simulation Engine
    def aircraft_dynamics(self, t, full_state):
        """Combine longitudinal and lateral dynamics"""
        state_long = full_state[:4]
        state_lat = full_state[4:]

        long_derivs = self.longitudinal_dynamics(t, state_long)
        lat_derivs = self.lateral_dynamics(t, state_lat)

        return np.array(long_derivs + lat_derivs)

    def run_simulation(self, duration=30, dt=0.1):
        """Run the aircraft simulation"""
        t_span = (0, duration)
        t_eval = np.arange(0, duration + dt, dt)

        solution = solve_ivp(self.aircraft_dynamics, t_span, self.state,
                             t_eval=t_eval, method='RK45')

        self.history['time'] = solution.t
        self.history['state'] = solution.y.T
        self.history['controls'] = np.tile(self.controls, (len(solution.t), 1))

        return solution

    def plot_results(self):
        """Plot simulation results"""
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))

        time = self.history['time']
        state = self.history['state']

        # Longitudinal plots
        axes[0, 0].plot(time, state[:, 0], 'b-')
        axes[0, 0].set_title('Forward Speed Perturbation')
        axes[0, 0].set_ylabel('u (m/s)')
        axes[0, 0].grid(True)

        axes[1, 0].plot(time, np.degrees(state[:, 1] / self.U0), 'r-')
        axes[1, 0].set_title('Angle of Attack')
        axes[1, 0].set_ylabel('α (degrees)')
        axes[1, 0].grid(True)

        axes[2, 0].plot(time, np.degrees(state[:, 3]), 'g-')
        axes[2, 0].set_title('Pitch Angle')
        axes[2, 0].set_ylabel('θ (degrees)')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].grid(True)

        # Lateral plots
        axes[0, 1].plot(time, np.degrees(state[:, 4] / self.U0), 'c-')
        axes[0, 1].set_title('Sideslip Angle')
        axes[0, 1].set_ylabel('β (degrees)')
        axes[0, 1].grid(True)

        axes[1, 1].plot(time, np.degrees(state[:, 5]), 'm-')
        axes[1, 1].set_title('Roll Rate')
        axes[1, 1].set_ylabel('p (deg/s)')
        axes[1, 1].grid(True)

        axes[2, 1].plot(time, np.degrees(state[:, 6]), 'y-')
        axes[2, 1].set_title('Yaw Rate')
        axes[2, 1].set_ylabel('r (deg/s)')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def analyze_modes(self):
        """Analyze aircraft dynamic modes (Phugoid, Short Period, Dutch Roll, etc.)"""
        print("Aircraft Dynamic Modes Analysis:")
        print("- Phugoid: Long-period pitch oscillation")
        print("- Short Period: Short-period pitch oscillation")
        print("- Dutch Roll: Lateral oscillation")
        print("- Roll Mode: Pure roll damping")
        print("- Spiral Mode: Slow divergence/convergence")


# Educational Missions - OUTSIDE CLASS
def run_educational_missions():
    """Run various flight scenarios to illustrate aircraft dynamics"""
    # Mission 1: Phugoid Oscillation
    print("Running Phugoid Oscillation Mission...")
    aircraft1 = AircraftSimulator()
    aircraft1.state = np.array([5.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    aircraft1.run_simulation(duration=60)
    aircraft1.plot_results()

    # Mission 2: Dutch Roll mode
    print("Mission 2: Dutch Roll Mode...")
    aircraft2 = AircraftSimulator()
    aircraft2.state = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.1, 0.0])
    aircraft2.run_simulation(duration=30)
    aircraft2.plot_results()


if __name__ == "__main__":
    aircraft = AircraftSimulator()
    aircraft.state = np.array([2.0, 0.5, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    aircraft.run_simulation(duration=20)
    aircraft.plot_results()
    aircraft.analyze_modes()

{
    "python.defaultInterpreterPath": "/Users/koltenpatton/aircraft-venv/bin/python"
}
