"Siddhant Desai - 1332240329 - S.Y. B.Sc. Physics - Semester III - Project Pased Learning"

#1 Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import constants
import time
from matplotlib.patches import Ellipse

#2 Declare Constants
G = constants.G      # Gravitational Constant (m^3.kg^-1.s^-2)
Rsun = 6.957e8       # Radius of the Sun (m)
Msun = 1.98847e30    # Mass of the Sun (Kg)
AU = constants.au    # Astronomical Unit
Day = 24 * 3600      # (s)
N = 1800             # Time Array
#3 Random number generator (based on current time)
rng = np.random.default_rng(int(time.time() % 2**31))

#4 Star and Planet Parameters
#4i Define Star Types
Star_Types = {'G': (0.9, 1.1), 'K': (0.7, 0.95), 'M': (0.2, 0.5)}   #(Radius Range, Mass Range)
#4ii Approximate Surface Temperature of Each Star Type
Tstar_dict = {'G': 5800, 'K': 4500, 'M': 3200}
#4iii Planet Parameters
Target_RP_RS = 0.20

#5 Geometry of the Planet Star Overlap
#5iCalculate the area of overlap between two circles (planet and star)
def overlap_area(d, Rs, Rp):
    if d >= Rs + Rp:
        return 0.0
    if d <= abs(Rs - Rp):
        return np.pi * min(Rs, Rp)**2
    Rs2, Rp2 = Rs**2, Rp**2
    alpha = np.arccos(np.clip((d**2 + Rs2 - Rp2) / (2 * d * Rs), -1, 1))
    beta = np.arccos(np.clip((d**2 + Rp2 - Rs2) / (2 * d * Rp), -1, 1))
    term = (-d + Rs + Rp) * (d + Rs - Rp) * (d - Rs + Rp) * (d + Rs + Rp)
    return Rs2 * alpha + Rp2 * beta - 0.5 * np.sqrt(max(0.0, term))
#5ii Finds Cosine of Angle Between Line of Sight and Stellar Surface Normal
def mu_from_pos(x, y, Rstar):
    r = np.sqrt(x**2 + y**2)
    if r >= Rstar:
        return 0.0
    return np.sqrt(1 - (r / Rstar)**2)

#6 Multi-run Simulation and Data Collection
#6i Creating Lists to Collect Data
Runs = 100  # Q1: 100 runs
star_types_list = []
Rstar_list = []
Mstar_list = []
Tstar_list = []
rp_over_rs_list = []
Rplanet_list = []
a_list = []
a_AU_list = []
Incl_deg_list = []
Incl_list = []
Impact_p_list = []
ulinear_list = []
Period_list = []
Period_days_list = []
Tduration_list = []
Tduration_Hours_list = []
TransitDepth_list = []
#6ii Generating the Star System
for run in range(Runs):
    #6iii Generate Star
    Stype = rng.choice(['G', 'K', 'M'], p=[0.45, 0.35, 0.20])
    Rstar = rng.uniform(*Star_Types[Stype]) * Rsun
    Mstar = rng.uniform(*Star_Types[Stype]) * Msun
    #6iv Approximate Surface Temperature of Each Star Type
    Tstar = Tstar_dict[Stype]
    #6v Planet to Star Radius Ratio
    rp_over_rs = Target_RP_RS * rng.uniform(0.85, 1.25)
    Rplanet = rp_over_rs * Rstar
    #6vi Radius of the Orbit (AU)
    a = rng.uniform(0.035, 0.08) * AU
    a_AU = a/AU   # Radius in Astronomical Units 
    #6vii Inclination (Degrees and Radians)
    Incl_deg = rng.uniform(87.5, 89.9)
    Incl = np.deg2rad(Incl_deg)
    #6viii Impact Parameter
    Impact_p = (a * np.cos(Incl))/Rstar
    Impact_p = min(Impact_p, 0.8)
    #6ix Linear Limb Darkening Coefficient
    ulinear = rng.uniform(0.35, 0.7)
    #6x Orbital Period Using Kepler's Third Law
    Period = 2 * np.pi * np.sqrt(a**3 / (G * Mstar))
    Period_days = Period/Day
    #6xi Transit Duration Calculation
    term = max(0.0, (1 + rp_over_rs)**2 - Impact_p**2)
    sinarg = ((Rstar/a) * np.sqrt(term)) / np.sin(Incl)
    sinarg = np.clip(sinarg, -1, 1)
    Tduration = (Period/np.pi) * np.arcsin(sinarg)
    Tduration_Hours = Tduration/3600
    #6xii Transit depth approximation (photometric)
    TransitDepth = (rp_over_rs**2)
    #6xiii Storing Run Data
    star_types_list.append(Stype)
    Rstar_list.append(Rstar)
    Mstar_list.append(Mstar)
    Tstar_list.append(Tstar)
    rp_over_rs_list.append(rp_over_rs)
    Rplanet_list.append(Rplanet)
    a_list.append(a)
    a_AU_list.append(a_AU)
    Incl_deg_list.append(Incl_deg)
    Incl_list.append(Incl)
    Impact_p_list.append(Impact_p)
    ulinear_list.append(ulinear)
    Period_list.append(Period)
    Period_days_list.append(Period_days)
    Tduration_list.append(Tduration)
    Tduration_Hours_list.append(Tduration_Hours)
    TransitDepth_list.append(TransitDepth)

#7 Data Conversion and Statistics
#7iConvert lists to numpy
rp_over_rs_arr = np.array(rp_over_rs_list)
TransitDepth_arr = np.array(TransitDepth_list)
a_AU_arr = np.array(a_AU_list)
Tduration_Hours_arr = np.array(Tduration_Hours_list)
Incl_deg_arr = np.array(Incl_deg_list)
Impact_p_arr = np.array(Impact_p_list)
ulinear_arr = np.array(ulinear_list)
Tstar_arr = np.array(Tstar_list)
#7ii Summary Statistics
mean_rp_rs = np.mean(rp_over_rs_arr)
mean_depth = np.mean(TransitDepth_arr)
mean_a_AU = np.mean(a_AU_arr)
mean_Tdur = np.mean(Tduration_Hours_arr)
median_rp_rs = np.median(rp_over_rs_arr)
median_depth = np.median(TransitDepth_arr)
#7iii Star type counts
unique_types, counts = np.unique(np.array(star_types_list), return_counts=True)
star_count_dict = dict(zip(unique_types, counts))
#7vi Correlations
corr_depth_rprs = np.corrcoef(TransitDepth_arr, rp_over_rs_arr)[0,1]
corr_Tdur_a = np.corrcoef(Tduration_Hours_arr, a_AU_arr)[0,1]

#8 Print Multi-run Summary
print("----- Multi-run Summary -----")
print(f"Number of simulated systems: {Runs}")
print(f"Average planet radius ratio (rp/rs): {mean_rp_rs:.4f}")
print(f"Median planet radius ratio (rp/rs): {median_rp_rs:.4f}")
print(f"Average transit depth: {mean_depth*100:.3f}%")
print(f"Median transit depth: {median_depth*100:.3f}%")
print(f"Average orbital radius: {mean_a_AU:.4f} AU")
print(f"Average transit duration: {mean_Tdur:.3f} hours")
print("Star type frequency:")
for t in ['G','K','M']:
    print(f"{t}: {star_count_dict.get(t,0)} systems")
print("---------------------------------")
print("Correlations (Pearson):")
print(f"Transit depth vs radius ratio: {corr_depth_rprs:.3f}")
print(f"Transit duration vs orbital radius: {corr_Tdur_a:.3f}")
print("---------------------------------")
if corr_depth_rprs > 0.7:
    print("Observation: The sample has consists of planets with masively varied radii.")
elif corr_depth_rprs > 0.3:
    print("Observation: The sample consists of planets with fairly different radius.")
else:
    print("Observation: Weak correlation between transit depth and radius ratio shos us that there is extremely low variance in the planet radii.")
if corr_Tdur_a > 0.3:
    print("Observation: Sample majorly consists of planets with larger orbits and transit duration.")
else:
    print("Observation: No clear trend is being followed due to the sample being dominated by short period orbits.")
print("---------------------------------")

#9 Trend Visualisations
def Bestfit(x, y):
    m, b = np.polyfit(x, y, 1)
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = m * x_fit + b
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label="Linear Best Fit")
#9i Transit Depth vs Radius Ratio
plt.figure(figsize=(6,4))
plt.scatter(rp_over_rs_arr, TransitDepth_arr*100, s=30, alpha=0.8)
Bestfit(rp_over_rs_arr,TransitDepth_arr*100)
plt.xlabel("Planet-to-Star Radius Ratio (rp/rs)")
plt.ylabel("Transit Depth [%]")
plt.title("Transit Depth vs Radius Ratio")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
#9ii Transit Duration vs Orbital Radius
plt.figure(figsize=(6,4))
plt.scatter(a_AU_arr, Tduration_Hours_arr, s=30, alpha=0.8)
Bestfit(a_AU_arr,Tduration_Hours_arr)
plt.xlabel("Orbital Radius [AU]")
plt.ylabel("Transit Duration [hours]")
plt.title("Transit Duration vs Orbital Radius")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
#9iii Transit Depth distribution
plt.figure(figsize=(6,4))
plt.hist(TransitDepth_arr*100, bins=18, edgecolor='black')
plt.xlabel("Transit Depth [%]")
plt.title("Distribution of Transit Depths")
plt.grid(alpha=0.15)
plt.tight_layout()
plt.show()
#9iv Planet radius distribution
plt.figure(figsize=(6,4))
plt.hist(rp_over_rs_arr, bins=18, edgecolor='black')
plt.xlabel("Planet-to-Star Radius Ratio (rp/rs)")
plt.title("Distribution of Planet Radius Ratios")
plt.grid(alpha=0.15)
plt.tight_layout()
plt.show()
#9v Star type frequency
plt.figure(figsize=(5,4))
types = ['G','K','M']
counts_plot = [star_count_dict.get(t,0) for t in types]
plt.bar(types, counts_plot)
plt.xlabel("Star Type")
plt.ylabel("Number of Systems")
plt.title("Star Type Frequency")
plt.grid(axis='y', alpha=0.15)
plt.tight_layout()
plt.show()
#9vi Inclination vs Impact Parameter
plt.figure(figsize=(6,4))
plt.scatter(Incl_deg_arr, Impact_p_arr, s=30, alpha=0.8)
Bestfit(Incl_deg_arr,Impact_p_arr)
plt.xlabel("Inclination [deg]")
plt.ylabel("Impact Parameter")
plt.title("Inclination vs Impact Parameter")
plt.grid(alpha=0.15)
plt.tight_layout()
plt.show()
#9vii Temperature vs Limb Darkening Coefficient
plt.figure(figsize=(6,4))
plt.scatter(Tstar_arr, ulinear_arr, s=30, alpha=0.8)
plt.xlabel("Stellar Temperature [K]")
plt.ylabel("Linear Limb Darkening Coefficient (u)")
plt.title("Temperature vs Limb Darkening")
plt.grid(alpha=0.15)
plt.tight_layout()
plt.show()

#10 Displaying a Single Transit and Star-Planet System
#10i Extract Data of the Last Run
Stype = star_types_list[-1]
Rstar = Rstar_list[-1]
Mstar = Mstar_list[-1]
Tstar = Tstar_list[-1]
rp_over_rs = rp_over_rs_list[-1]
Rplanet = Rplanet_list[-1]
a = a_list[-1]
a_AU = a_AU_list[-1]
Incl_deg = Incl_deg_list[-1]
Incl = Incl_list[-1]
Impact_p = Impact_p_list[-1]
ulinear = ulinear_list[-1]
Period = Period_list[-1]
Period_days = Period_days_list[-1]
Tduration = Tduration_list[-1]
Tduration_Hours = Tduration_Hours_list[-1]
#10ii Time Array for the Last Run
t = np.linspace(-1.5 * Tduration, 1.5 * Tduration, N)
t_hours = t / 3600
#10iii Generating a Single Transit Light Curve
I_disk_avg = 1 - (ulinear/3)    # Average Stellar Intensity
x = a * (t / (Tduration * 2))   # linear motion across stellar disk
y = Impact_p * Rstar
d = np.sqrt(x**2 + y**2)        # distance between centers
#10iv Flux Calculation
flux = np.zeros_like(t)
for i in range(N):
    Aoverlap = overlap_area(d[i], Rstar, Rplanet)
    mu = mu_from_pos(x[i], y, Rstar)
    Ic = 1 - ulinear * (1 - mu)
    flux[i] = 1 - (Aoverlap / (np.pi * Rstar**2)) * (Ic / I_disk_avg)
#10v Gaussian Noise
Noise_signal = 0.0025
Smooth_signal = 4
noisy_flux = flux + rng.normal(0, Noise_signal, flux.shape)
smoothed_flux = gaussian_filter1d(noisy_flux, sigma=Smooth_signal)

#11 Results of the Last Run
print("-------Randomized Exoplanet System (LAST RUN)-------")
print(f"Star type: {Stype}")
print(f"Radius of the Star = {Rstar/Rsun:.3f} Solar Radius")
print(f"Mass of the Star = {Mstar/Msun:.3f} Solar Mass")
print(f"Temperature of the Star ≈ {Tstar}K")
print(f"Planet-to-Star radius ratio= {rp_over_rs:.3f}")
print(f"Orbital radius = {a_AU:.4f} AU")
print(f"Inclination = {Incl_deg:.2f}°")
print(f"Transit duration = {Tduration_Hours:.2f} hours")
print(f"Transit depth ≈ {(rp_over_rs**2)*100:.2f}%")
print(f"Limb darkening coefficient u = {ulinear:.3f}")
print("---------------------------------------------------")

#12 Plotting the Single Transit of the Last Run
plt.figure(figsize=(9, 4))
plt.scatter(t_hours, noisy_flux, s=8, alpha=0.6, label="Observed (noisy)")
plt.plot(t_hours, smoothed_flux, "-", lw=2, color="red", label="Smoothed model")
plt.xlabel("Time [hours]")
plt.ylabel("Normalized Flux")
plt.title("Simulated Exoplanet Transit Light Curve (LAST RUN)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

#13 Orbit Visualization
#13i Star Color
Star_colors = {"G": "gold", "K": "orange", "M": "red"}
Scolor = Star_colors[Stype]
#13ii Background
fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
ax.set_facecolor('black')
ax.set_aspect('equal')
#13iii Generate Star
star = plt.Circle((0, 0), Rstar, color=Scolor, ec='none', zorder=3)
ax.add_patch(star)
#13iv Generate Orbit Path
orbit = Ellipse((0, 0), width=2*a, height=2*a*0.6, angle=0,
                edgecolor='white', ls='--', facecolor='none', alpha=0.4)
ax.add_patch(orbit)
#13v Planet Positions Along The Orbit
num_positions = 12
angles = np.linspace(0, 2 * np.pi, num_positions, endpoint=False)
for angle in angles:
    x_p = a * np.cos(angle)
    y_p = a * 0.6 * np.sin(angle)
    trail = plt.Circle((x_p, y_p), Rplanet * 0.8, color='deepskyblue', alpha=0.5)
    ax.add_patch(trail)
#13vi Highlight Current Planet Position
planet_now = plt.Circle((a, 0), Rplanet, color='deepskyblue', ec='white', lw=1.0)
ax.add_patch(planet_now)
#13vii Define Axis Limits
ax.set_xlim(-1.3 * a, 1.3 * a)
ax.set_ylim(-1.3 * a, 1.3 * a)
ax.axis('off')
#12viii Plot
plt.title(f"Orbit Around a {Stype}-Type Star (LAST RUN)", color='white', pad=15)
plt.tight_layout()
plt.show()
"End of Code"