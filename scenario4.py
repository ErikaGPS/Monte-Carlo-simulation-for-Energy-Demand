import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
from scipy.stats import skewnorm
import random

np.random.seed(42)

# Constants for Ammonia-based Energy Demand Simulation (Including Nitrogen Separation Energy)
PRIMARY_ENERGY = 3.7361146900807e11 * 2.7777777777778E-10  # MJ to kWh
ITERATIONS = 100000
NITROGEN_NH3 = 0.823    # kg N2 per kg NH3
HYDROGEN_NH3 = 0.176    # kg H2 per kg NH3
LHV_NH3 = 5.17          # kWh / kg
LHV_H2 = 33.333         # kWh / kg
H2O_H2 = 9              # kg H2O / kg H2

# Scenario parameter dictionaries, separated into optimistic, most likely, and pessimistic ranges

optimistic_params = {"DIESEL": (47, 52), "AMMONIA_ICE": (38, 45), "HABER_BOSCH": (60, 70),
                     "ELECTROLYSIS": (65, 70), "DESALINATION": (0.004, 0.003), "SEPARATION": (0.2, 0.11), "STORAGE": (94, 95), "VAPORIZATION": (97, 98),
                     "NITROGEN_NH3": NITROGEN_NH3, "HYDROGEN_NH3": HYDROGEN_NH3, "H2O_H2": H2O_H2, "LHV_NH3": LHV_NH3, "LHV_H2": LHV_H2}

most_likely_params = {"DIESEL": (40, 47), "AMMONIA_ICE": (30, 38), "HABER_BOSCH": (50, 60),
                      "ELECTROLYSIS": (60, 65), "DESALINATION": (0.005, 0.004), "SEPARATION": (0.28, 0.20), "STORAGE": (92, 94), "VAPORIZATION": (96, 97),
                      "NITROGEN_NH3": NITROGEN_NH3, "HYDROGEN_NH3": HYDROGEN_NH3, "H2O_H2": H2O_H2, "LHV_NH3": LHV_NH3, "LHV_H2": LHV_H2}

pessimistic_params = {"DIESEL": (35, 40), "AMMONIA_ICE": (22, 30), "HABER_BOSCH": (40, 50),
                      "ELECTROLYSIS": (55, 60), "DESALINATION": (0.007, 0.005), "SEPARATION": (0.37, 0.28), "STORAGE": (90, 92), "VAPORIZATION": (95, 96),
                      "NITROGEN_NH3": NITROGEN_NH3, "HYDROGEN_NH3": HYDROGEN_NH3, "H2O_H2": H2O_H2, "LHV_NH3": LHV_NH3, "LHV_H2": LHV_H2}

# Functions for energy calculations

# Step 1: Calculate useful energy (energy available for diesel propulsion) from primary energy
def useful_energy(diesel_min, diesel_max, primary_energy):
    # propulsion energy for diesel engine
    efficiency = np.random.uniform(diesel_min, diesel_max) / 100 # yields efficiency as fraction
    return primary_energy * efficiency


# Step 2: Calculate ammonia energy demand/requirement from useful energy
def ammonia_energy_demand(useful_energy, engine_min, engine_max):
    # propulsion energy requirement for ammonia engine
    efficiency = np.random.uniform(engine_min, engine_max) / 100
    return useful_energy / efficiency

# Calculate Ammonia energy in mass using Lower Heating Value of NH3 = 5.17 kWh / kg
def ammonia_mass(ammonia_energy_demand):
    return ammonia_energy_demand / LHV_NH3

# Step 3: Calculate Hydrogen demand for Ammonia Synthesis

# 3.1: Find hydrogen mass per unit ammonia = 0.176 kg H2 / kg NH3
def hydrogen_mass(ammonia_mass):
    return ammonia_mass * HYDROGEN_NH3

# 3.2: Find electrolysis energy to produce hydrogen mass
def hydrogen_electrolysis_energy(hydrogen_mass, electrolysis_min, electrolysis_max):
    electrolysis_eff = np.random.uniform(electrolysis_min, electrolysis_max) / 100 # fraction
    return hydrogen_mass * (LHV_H2 / electrolysis_eff) # converts electrolysis energy from fraction to kWh

# 3.3: Calculate water requirement for hydrogen production with water-to-hydrogen ratio (9 kg H2O / kg H2)
def water_mass_requirement(hydrogen_mass):
    return hydrogen_mass * H2O_H2

# 3.4: Find water purification energy for water requirement
def water_desalination(water_mass_requirement, desalination_min, desalination_max):
    desalination_eff = np.random.uniform(desalination_min, desalination_max) # given in kWh / kg H2O
    return water_mass_requirement * desalination_eff

# Step 4: Calculate Nitrogen demand for Ammonia Synthesis

# 4.1: Required nitrogen mass per unit ammonia = 0.823 kg N2 / kg NH3
def nitrogen_mass(ammonia_mass):
    return ammonia_mass * NITROGEN_NH3

# 4.2: Use required nitrogen mass to find air separation energy
def nitrogen_separation_energy(nitrogen_mass, separation_min, separation_max):
    separation_eff = np.random.uniform(separation_min, separation_max) # in kWh / kg N2
    return nitrogen_mass * separation_eff


# Step 5: Calculate energy requirement to synthesize ammonia via Haber Bosch
def haber_bosch(ammonia_mass, haberbosch_min, haberbosch_max):
    haberbosch_eff = np.random.uniform(haberbosch_min, haberbosch_max) / 100
    return ammonia_mass * (LHV_NH3 / haberbosch_eff) # converts HB energy from fraction to kWh

# Step 6: Calculate total energy requirement for ammonia production (total of all process steps)
def total_synthesis_energy(hydrogen_electrolysis_energy, water_desalination, nitrogen_separation_energy, haber_bosch):
    return hydrogen_electrolysis_energy + water_desalination + nitrogen_separation_energy + haber_bosch

# Step 7: Consider storage and vaporization losses
def final_energy(total_synthesis_energy, storage_min, storage_max, vaporization_min, vaporization_max):
    storage_eff = np.random.uniform(storage_min, storage_max) / 100 # treated as fraction
    vaporization_eff = np.random.uniform(vaporization_min, vaporization_max) / 100 # fraction
    return total_synthesis_energy / (storage_eff * vaporization_eff)



# Function of Monte Carlo simulation for scenario ranges. Index 0 = min. value of range, 1 = max. value of range

def run_scenario(params, primary_energy, iterations=ITERATIONS): # Array to store energy quantities for each process
    energy_results = np.zeros(iterations)
    for i in range(iterations):
        ue = useful_energy(params["DIESEL"][0], params["DIESEL"][1], primary_energy)
        ae = ammonia_energy_demand(ue, params["AMMONIA_ICE"][0], params["AMMONIA_ICE"][1])
        am = ammonia_mass(ae)
        hm = hydrogen_mass(am)
        he = hydrogen_electrolysis_energy(hm, params["ELECTROLYSIS"][0], params["ELECTROLYSIS"][1])
        wm = water_mass_requirement(hm)
        wd = water_desalination(wm, params["DESALINATION"][0], params["DESALINATION"][1])
        nm = nitrogen_mass(am)
        ns = nitrogen_separation_energy(nm, params["SEPARATION"][0], params["SEPARATION"][1])
        hb = haber_bosch(am, params["HABER_BOSCH"][0], params["HABER_BOSCH"][1])
        ts = total_synthesis_energy(he, wd, ns, hb)
        final = final_energy(ts, params["STORAGE"][0], params["STORAGE"][1], params["VAPORIZATION"][0], params["VAPORIZATION"][1])
        energy_results[i] = final
    return energy_results

# Run simulations in separate scenarios, defined by x_parameter range.

energy_optimistic = run_scenario(optimistic_params, PRIMARY_ENERGY, ITERATIONS)
energy_most_likely = run_scenario(most_likely_params, PRIMARY_ENERGY, ITERATIONS)
energy_pessimistic = run_scenario(pessimistic_params, PRIMARY_ENERGY, ITERATIONS)


# Compute mean and std for each scenario
mean_opt, std_opt = np.mean(energy_optimistic), np.std(energy_optimistic)
mean_most_likely, std_most_likely = np.mean(energy_most_likely), np.std(energy_most_likely)
mean_pess, std_pess = np.mean(energy_pessimistic), np.std(energy_pessimistic)


# Fit skew-normal distributions to each energy data
# a = shape / skewness , loc = location (mean) , scale = standard deviation
a_opt, loc_opt, scale_opt = skewnorm.fit(energy_optimistic)
a_most_likely, loc_most_likely, scale_most_likely = skewnorm.fit(energy_most_likely)
a_pess, loc_pess, scale_pess = skewnorm.fit(energy_pessimistic)


# Estimate bins from Friedman Diaconis rule: bin width -> 2 * IQR / (iterations ** 1/3)
# IQR (Interquartile range) is the difference between the 0.25 and 0.75 percentiles of the data set
# Number of bins to span full range = ( max(data range) - min(data range) ) / bin width

# Optimistic bins
OQ1 = np.quantile(energy_optimistic, 0.25)
OQ2 = np.quantile(energy_optimistic, 0.75)
IQR = OQ2 - OQ1
O_bin= 2 * IQR / (ITERATIONS**(1/3))
print((max(energy_optimistic) - min(energy_optimistic)) / O_bin) # printed 96.3

# Most likely bins
MQ1 = np.quantile(energy_most_likely, 0.25)
MQ2 = np.quantile(energy_most_likely, 0.75)
MIQR = MQ2 - MQ1
M_bin= 2 * MIQR / (ITERATIONS**(1/3))
print((max(energy_most_likely) - min(energy_most_likely)) / M_bin) # printed 94

# Pessimistic bins
PQ1 = np.quantile(energy_pessimistic, 0.25)
PQ2 = np.quantile(energy_pessimistic, 0.75)
PIQR = PQ2 - PQ1
P_bin= 2 * PIQR / (ITERATIONS**(1/3))
print((max(energy_pessimistic) - min(energy_pessimistic)) / P_bin) # printed 88

# Bins must be equal for each scenario - use 100 bins

bin_interval1 = np.linspace(energy_optimistic.min(), energy_optimistic.max(), 100)
bin_interval2 = np.linspace(energy_most_likely.min(), energy_most_likely.max(), 100)
bin_interval3 = np.linspace(energy_pessimistic.min(), energy_pessimistic.max(), 100)


# Computing center of each bin (x-coordinates) to get a smooth probability density function (PDF).

midpoints1 = (bin_interval1[:-1] + bin_interval1[1:]) / 2
midpoints2 = (bin_interval2[:-1] + bin_interval2[1:]) / 2
midpoints3 = (bin_interval3[:-1] + bin_interval3[1:]) / 2

# Compute PDF at midpoints (x). Parameters: a = skewness (if a = 0 -> normal PDF), loc = mean, scale = st.dev

skewed_pdf_opt = skewnorm.pdf(midpoints1, a_opt, loc_opt, scale_opt)
skewed_pdf_most_likely = skewnorm.pdf(midpoints2, a_most_likely, loc_most_likely, scale_most_likely)
skewed_pdf_pess = skewnorm.pdf(midpoints3, a_pess, loc_pess, scale_pess)

# Find how many counts in each bin. Numpy histogram returns counts + bin edges (but use own definition, bin_interval)

counts_opt, _ = np.histogram(energy_optimistic, bins=bin_interval1)
counts_most, _ = np.histogram(energy_most_likely, bins=bin_interval2)
counts_pess, _ = np.histogram(energy_pessimistic, bins=bin_interval3)



# PLOTS OF REQUIRED ELECTRICITY INPUT (TWH) FOR AMMONIA PRODUCTION + SKEWED GAUSSIAN FIT

# Total annual electricity output of Norway's renewable energy system (2024)
today = 157 # TWh
def twh_to_percentage(x):
    return (x / today) * 100


# OPTIMISTIC

plt.figure(figsize=(10, 6))
plt.bar(midpoints1, counts_opt, width=np.diff(bin_interval1)[0], color = "yellowgreen", edgecolor="black")
plt.hist(energy_optimistic, bins=bin_interval1, alpha=0.4, color="forestgreen", label=f"Optimistic Scenario")
plt.plot(midpoints1, skewed_pdf_opt * len(energy_optimistic) * np.diff(bin_interval1)[0], color="darkgreen", linestyle="--", label="Optimistic Fit")
plt.axvline(x=mean_opt, color="black", linestyle="--", label=f"Mean (μ) = {mean_opt:.2f} TWh")
plt.axvline(x=mean_opt-std_opt, color="orange", linestyle="--", label=f"μ - 1σ = {mean_opt-std_opt:.2f} TWh")
plt.axvline(x=mean_opt+std_opt, color="orange", linestyle="--", label=f"μ + 1σ = {mean_opt+std_opt:.2f} TWh")
plt.axvline(x=mean_opt - 2 * std_opt, color="red", linestyle="--", label=f"μ - 2σ = {mean_opt - 2 * std_opt:.2f} TWh")
plt.axvline(x=mean_opt + 2 * std_opt, color="red", linestyle="--", label=f"μ + 2σ = {mean_opt + 2 * std_opt:.2f} TWh")
plt.fill_betweenx(y=[0, max(counts_most)], x1=mean_opt-std_opt, x2=mean_opt+std_opt, color ="lightgreen", alpha=0.3)
plt.xlabel("Energy Intervals (TWh)", fontsize=14)
plt.xlim(0,1000)
plt.ylabel("Frequency of Simulated Energy Demand", fontsize=14)
plt.title("Energy Optimistic Scenario for Ammonia-Based Shipping with Skewed Gaussian Fit", fontsize=16)
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5), fontsize=14)
y_top = plt.ylim()[1]
y_offset = 0.05 * y_top  # adjust vertical offset as needed
plt.text(mean_opt*1.15, y_top - y_offset, f"Mean = {twh_to_percentage(mean_opt):.1f}% of Norway's \n renewable electricity \n generation (2024)",
        ha="left", va="top", fontsize=14, color="black")
plt.gca().text(0.01, 0.01, '(a)', transform=plt.gca().transAxes, fontsize=14, fontweight='bold',va='bottom', ha='left')
plt.tight_layout()

# MOST LIKELY

plt.figure(figsize=(10, 6))
plt.bar(midpoints2, counts_most, width=np.diff(bin_interval2)[0], color = "skyblue", edgecolor="black")
plt.hist(energy_most_likely, bins=bin_interval2, alpha=0.4, color="skyblue", label=f"Most Likely Scenario")
plt.plot(midpoints2, skewed_pdf_most_likely * len(energy_most_likely) * np.diff(bin_interval2)[0], color="darkblue", linestyle="--", label="Most Likely Fit")
plt.axvline(x=mean_most_likely, color="black", linestyle="--", label=f"Mean (μ) = {mean_most_likely:.2f} TWh")
plt.axvline(x=mean_most_likely-std_most_likely, color="orange", linestyle="--", label=f"μ - 1σ = {mean_most_likely-std_most_likely:.2f} TWh")
plt.axvline(x=mean_most_likely+std_most_likely, color="orange", linestyle="--", label=f"μ + 1σ = {mean_most_likely+std_most_likely:.2f} TWh")
plt.axvline(x=mean_most_likely - 2 * std_most_likely, color="red", linestyle="--", label=f"μ - 2σ = {mean_most_likely - 2 * std_most_likely:.2f} TWh")
plt.axvline(x=mean_most_likely + 2 * std_most_likely, color="red", linestyle="--", label=f"μ + 2σ = {mean_most_likely + 2 * std_most_likely:.2f} TWh")
plt.fill_betweenx(y=[0, max(counts_most)], x1=mean_most_likely-std_most_likely, x2=mean_most_likely+std_most_likely, color ="lightblue", alpha=0.3)
plt.xlabel("Energy Intervals (TWh)", fontsize=14)
plt.xlim(0,1000)
plt.ylabel("Frequency of Simulated Energy Demand", fontsize=14)
plt.title("Energy Most Likely Scenario for Ammonia-Based Shipping with Skewed Gaussian Fit", fontsize=16)
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5), fontsize=14)
y_top = plt.ylim()[1]
y_offset = 0.05 * y_top  # adjust vertical offset as needed
plt.text(mean_most_likely*1.025, y_top - y_offset, f"Mean = {twh_to_percentage(mean_most_likely):.1f}% of Norway's \n renewable electricity \n generation (2024)",
        ha="left", va="top", fontsize=14, color="black")
plt.gca().text(0.01, 0.01, '(b)', transform=plt.gca().transAxes, fontsize=14, fontweight='bold',va='bottom', ha='left')
plt.tight_layout()
plt.show()


# PESSIMISTIC

plt.figure(figsize=(10, 6))
plt.bar(midpoints3, counts_pess, width=np.diff(bin_interval3)[0], color= "firebrick", edgecolor="black")
plt.hist(energy_pessimistic, bins=bin_interval3, alpha=0.4, color="indianred", label=f"Pessimistic Scenario")
plt.plot(midpoints3, skewed_pdf_pess * len(energy_pessimistic) * np.diff(bin_interval3)[0], color="darkred", linestyle="--", label="Pessimistic Fit")
plt.axvline(x=mean_pess, color="black", linestyle="--", label=f"Mean (μ)= {mean_pess:.2f} TWh")
plt.axvline(x=mean_pess-std_pess, color="orange", linestyle="--", label=f"μ - 1σ = {mean_pess-std_pess:.2f} TWh")
plt.axvline(x=mean_pess+std_pess, color="orange", linestyle="--", label=f"μ + 1σ = {mean_pess+std_pess:.2f} TWh")
plt.axvline(x=mean_pess - 2 * std_pess, color="red", linestyle="--", label=f"μ - 2σ = {mean_pess - 2 * std_pess:.2f} TWh")
plt.axvline(x=mean_pess + 2 * std_pess, color="red", linestyle="--", label=f"μ + 2σ = {mean_pess + 2 * std_pess:.2f} TWh")
plt.fill_betweenx(y=[0, max(counts_pess)], x1=mean_pess-std_pess, x2=mean_pess+std_pess, color ="pink", alpha=0.3)
plt.xlabel("Energy Intervals (TWh)", fontsize=14)
plt.xlim(0,1000)
plt.ylabel("Frequency of Simulated Energy Demand", fontsize=14)
plt.title("Energy Pessimistic Scenario for Ammonia-Based Shipping with Skewed Gaussian Fit", fontsize=16)
plt.legend(loc = "center left",fontsize=14)
y_top = plt.ylim()[1]
y_offset = 0.05 * y_top
plt.text(mean_pess*1.025, y_top - y_offset, f"Mean = {twh_to_percentage(mean_pess):.1f}% of Norway's \n renewable electricity \n generation (2024)",
        ha="left", va="top", fontsize=14, color="black")
plt.gca().text(0.01, 0.01, '(c)', transform=plt.gca().transAxes, fontsize=14, fontweight='bold',va='bottom', ha='left')
plt.tight_layout()
plt.show()


# MONTE CARLO SIMULATION FOR RENEWABLE ENERGY SOURCES

# Define Capacity Factor ranges
RENEWABLES_OPT = {"ONSHORE": (40, 45), "OFFSHORE": (50, 56.5), "SOLAR": (20,25), "HYDRO": (44, 50)}
RENEWABLES_MOST = {"ONSHORE": (35, 40), "OFFSHORE": (45, 50), "SOLAR": (15, 20), "HYDRO": (36, 44)}
RENEWABLES_PESS = {"ONSHORE": (30, 35), "OFFSHORE": (40, 45), "SOLAR": (9, 15), "HYDRO": (30, 36)}

# 8760 number of hours in a year
# Multiply final energy with 1e6 to convert from TWh to MWh

def offshore_wind(offshore_min, offshore_max, final_energy):
    # Capacity factor (as a fraction)
    CF_offshore = np.random.uniform(offshore_min, offshore_max) / 100
    # Required installed capacity MW
    return final_energy * 1e6 / (CF_offshore * 8760), CF_offshore

def onshore_wind(onshore_min, onshore_max, final_energy):
    CF_onshore = np.random.uniform(onshore_min, onshore_max) / 100
    return final_energy * 1e6 / (CF_onshore * 8760), CF_onshore


def solar_PV(solar_min, solar_max, final_energy):
    CF_solar = np.random.uniform(solar_min, solar_max) / 100
    return final_energy * 1e6 / (CF_solar * 8760), CF_solar

def hydro_power(hydro_min, hydro_max, final_energy):
    CF_hydro = np.random.uniform(hydro_min, hydro_max) / 100
    return final_energy * 1e6 / (CF_hydro * 8760), CF_hydro

# Units are in MWh. Converted final energy in TWh to MWh by multiplying with 1e6

def run_scenario2(renewables, energy_array): # Array to store capacities for each source
    iterations = len(energy_array)
    MW_results = np.zeros((iterations, 8)) # Columns: [Offshore, CF_off, Onshore, CF_on, Solar, CF_solar, Hydro, CF_hydro]
    for i in range(iterations): # 100.000
        final_energy = energy_array[i]
        offshore, cf_offshore = offshore_wind(renewables["OFFSHORE"][0], renewables["OFFSHORE"][1], final_energy)
        onshore, cf_onshore = onshore_wind(renewables["ONSHORE"][0], renewables["ONSHORE"][1], final_energy)
        solar, cf_solar = solar_PV(renewables["SOLAR"][0], renewables["SOLAR"][1], final_energy)
        hydro, cf_hydro = hydro_power(renewables["HYDRO"][0], renewables["HYDRO"][1], final_energy)
        MW_results[i] = [offshore, cf_offshore, onshore, cf_onshore, solar, cf_solar, hydro, cf_hydro]
    return MW_results

# Run scenario 2 with scenario 1

MW_results_opt = run_scenario2(RENEWABLES_OPT, energy_optimistic)
MW_results_most = run_scenario2(RENEWABLES_MOST, energy_most_likely)
MW_results_pess = run_scenario2(RENEWABLES_PESS, energy_pessimistic)


# Calculate mean and standard deviation for each scenario

mean_off, std_off = np.mean(MW_results_opt[:,0]), np.std(MW_results_opt[:,0])
mean_off2, std_off2 = np.mean(MW_results_most[:,0]), np.std(MW_results_most[:,0])
mean_off3, std_off3 = np.mean(MW_results_pess[:,0]), np.std(MW_results_pess[:,0])

mean_on, std_on = np.mean(MW_results_opt[:,2]), np.std(MW_results_opt[:,2])
mean_on2, std_on2 = np.mean(MW_results_most[:,2]), np.std(MW_results_most[:,2])
mean_on3, std_on3 = np.mean(MW_results_pess[:,2]), np.std(MW_results_pess[:,2])

mean_sol, std_sol = np.mean(MW_results_opt[:,4]), np.std(MW_results_opt[:,4])
mean_sol2, std_sol2 = np.mean(MW_results_most[:,4]), np.std(MW_results_most[:,4])
mean_sol3, std_sol3 = np.mean(MW_results_pess[:,4]), np.std(MW_results_pess[:,4])

mean_hyd, std_hyd = np.mean(MW_results_opt[:,6]), np.std(MW_results_opt[:,6])
mean_hyd2, std_hyd2 = np.mean(MW_results_most[:,6]), np.std(MW_results_most[:,6])
mean_hyd3, std_hyd3 = np.mean(MW_results_pess[:,6]), np.std(MW_results_pess[:,6])

bins = 100  # Using the same number of bins for consistency


# OPTIMISTIC INSTALLED CAPACITY (MW). Index = 0 (offshore) ; 2 (onshore) ; 4 (solar) ; 6 (hydro) in Monte Carlo

plt.figure(figsize=(10,6))
plt.xscale('symlog', linthresh=1e4)
plt.hist(MW_results_opt[:,0], bins=bins, color="skyblue", alpha=0.7, label="Offshore wind")
plt.axvline(x=mean_off, color="black", linestyle="solid", label=f"Offshore μ ± 1σ = {mean_off:,.0f} ± {std_off:,.0f} MW")
plt.hist(MW_results_opt[:,2], bins=bins, color="forestgreen", alpha=0.7, label="Onshore wind")
plt.axvline(x=mean_on, color="black", linestyle="dotted", label=f"Onshore μ ± 1σ = {mean_on:,.0f} ± {std_on:,.0f} MW")
plt.hist(MW_results_opt[:,4], bins=bins, color="gold", alpha=0.7, label="Solar PV")
plt.axvline(x=mean_sol, color="black", linestyle="dashed", label=f"Solar PV μ ± 1σ = {mean_sol:,.0f} ± {std_sol:,.0f} MW")
plt.hist(MW_results_opt[:,6], bins=bins, color="firebrick", alpha=0.7, label="Hydro Power")
plt.axvline(x=mean_hyd, color="black", linestyle="dashdot", label=f"Hydro Power μ ± 1σ = {mean_hyd:,.0f} ± {std_hyd:,.0f} MW")
plt.xlabel("Installed Capacity (MW)", fontsize=14)
plt.xlim(70000,1200000)
plt.ylabel("Number of Observations", fontsize=14)
plt.title("Renewable Installed Capacity Distributions (Optimistic Scenario)", fontsize=16)
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5), fontsize=13)
plt.gca().text(0.01, 0.01, '(a)', transform=plt.gca().transAxes, fontsize=14, fontweight='bold',va='bottom', ha='left')
import matplotlib.ticker as ticker
ax = plt.gca()
ax.tick_params(axis='x', labelsize=14)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune='both'))
plt.tight_layout()
plt.show()



#MOST LIKELY INSTALLED CAPACITY (MW). Index = 0 (offshore) ; 2 (onshore) ; 4 (solar) ; 6 (hydro) in Monte Carlo

plt.figure(figsize=(10,6))
plt.xscale('symlog', linthresh=1e4)
plt.hist(MW_results_most[:,0], bins=bins, color="skyblue", alpha=0.7, label="Offshore wind")
plt.axvline(x=mean_off2, color="black", linestyle="solid", label=f"Offshore μ ± 1σ = {mean_off2:,.0f} ± {std_off2:,.0f} MW")
plt.hist(MW_results_most[:,2], bins=bins, color="forestgreen", alpha=0.7, label="Onshore wind")
plt.axvline(x=mean_on2, color="black", linestyle="dotted", label=f"Onshore μ ± 1σ = {mean_on2:,.0f} ± {std_on2:,.0f} MW")
plt.hist(MW_results_most[:,4], bins=bins, color="gold", alpha=0.7, label="Solar PV")
plt.axvline(x=mean_sol2, color="black", linestyle="dashed", label=f"Solar PV μ ± 1σ = {mean_sol2:,.0f} ± {std_sol2:,.0f} MW")
plt.hist(MW_results_most[:,6], bins=bins, color="firebrick", alpha=0.7, label="Hydro Power")
plt.axvline(x=mean_hyd2, color="black", linestyle="dashdot", label=f"Hydro Power μ ± 1σ = {mean_hyd2:,.0f} ± {std_hyd2:,.0f} MW")
plt.xlabel("Installed Capacity (MW)", fontsize=14)
plt.xlim(70000,1200000)
plt.ylabel("Number of Observations", fontsize=14)
plt.title("Renewable Installed Capacity Distributions (Most Likely Scenario)", fontsize=16)
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5), fontsize=13)
plt.gca().text(0.01, 0.01, '(b)', transform=plt.gca().transAxes, fontsize=14, fontweight='bold',va='bottom', ha='left')
ax = plt.gca()
ax.tick_params(axis='x', labelsize=14)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune='both'))
plt.tight_layout()
plt.show()


# PESSIMISTIC INSTALLED CAPACITY (MW). Index = 0 (offshore) ; 2 (onshore) ; 4 (solar) ; 6 (hydro) in Monte Carlo

plt.figure(figsize=(10,6))
plt.xscale('symlog', linthresh=1e4)
plt.hist(MW_results_pess[:,0], bins=bins, color="skyblue", alpha=0.7, label="Offshore wind")
plt.axvline(x=mean_off3, color="black", linestyle="solid", label=f"Offshore μ ± 1σ = {mean_off3:,.0f} ± {std_off3:,.0f} MW")
plt.hist(MW_results_pess[:,2], bins=bins, color="forestgreen", alpha=0.7, label="Onshore wind")
plt.axvline(x=mean_on3, color="black", linestyle="dotted", label=f"Onshore μ ± 1σ = {mean_on3:,.0f} ± {std_on3:,.0f} MW")
plt.hist(MW_results_pess[:,4], bins=bins, color="gold", alpha=0.7, label="Solar PV")
plt.axvline(x=mean_sol3, color="black", linestyle="dashed", label=f"Solar PV μ ± 1σ = {mean_sol3:,.0f} ± {std_sol3:,.0f} MW")
plt.hist(MW_results_pess[:,6], bins=bins, color="firebrick", alpha=0.7, label="Hydro Power")
plt.axvline(x=mean_hyd3, color="black", linestyle="dashdot", label=f"Hydro Power μ ± 1σ = {mean_hyd3:,.0f} ± {std_hyd3:,.0f} MW")
plt.xlabel("Installed Capacity (MW)", fontsize=14)
plt.xlim(70000,1200000)
plt.ylabel("Number of Observations", fontsize=14)
plt.title("Renewable Installed Capacity Distributions (Pessimistic Scenario)", fontsize=16)
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5), fontsize=13)
plt.gca().text(0.01, 0.01, '(c)', transform=plt.gca().transAxes, fontsize=14, fontweight='bold',va='bottom', ha='left')
ax = plt.gca()
ax.tick_params(axis='x', labelsize=14)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune='both'))
plt.tight_layout()
plt.show()




# SPATIAL REQUIREMENTS

# Area (km2) = Installed Capacity (MW) / Capacity Density (MW / km2)

# Capacity Density (MW / km2) -  Offshore, Onshore, Solar, Hydro

Offshore_cap = 3.5 # Referenced
Onshore_cap = 8.6  # Referenced
Solar_cap = 77     # Calculated from reference (DOMMA planned solar park: 264,4 MWp distributed on 3,441 km2 land)
Hydro_cap = 27     # Calculated from reference (Total MW production / total occupied direct area)
# Hydro_cap1= 5.6  # Calculated from reference (Total MW production / total occupied direct + surrounding area)


# Offshore wind
area_offshore_opt = (MW_results_opt[:, 0]) / Offshore_cap
area_offshore_most = (MW_results_most[:, 0]) / Offshore_cap
area_offshore_pess = (MW_results_pess[:, 0]) / Offshore_cap

# Onshore wind
area_onshore_opt = (MW_results_opt[:, 2]) / Onshore_cap
area_onshore_most = (MW_results_most[:, 2]) / Onshore_cap
area_onshore_pess = (MW_results_pess[:, 2]) / Onshore_cap

# Solar
area_solar_opt = (MW_results_opt[:, 4]) / Solar_cap
area_solar_most = (MW_results_most[:, 4]) / Solar_cap
area_solar_pess = (MW_results_pess[:, 4]) / Solar_cap

# Hydro
area_hydro_opt = (MW_results_opt[:, 6]) / Hydro_cap
area_hydro_most = (MW_results_most[:, 6]) / Hydro_cap
area_hydro_pess = (MW_results_pess[:, 6]) / Hydro_cap


# Calculate MEAN and STD for area

# OFFSHORE
mean_area_off_opt, std_area_off_opt = np.mean(area_offshore_opt), np.std(area_offshore_opt)
mean_area_off_most, std_area_off_most = np.mean(area_offshore_most), np.std(area_offshore_most)
mean_area_off_pess, std_area_off_pess = np.mean(area_offshore_pess), np.std(area_offshore_pess)
# ONSHORE
mean_area_on_opt, std_area_on_opt = np.mean(area_onshore_opt), np.std(area_onshore_opt)
mean_area_on_most, std_area_on_most = np.mean(area_onshore_most), np.std(area_onshore_most)
mean_area_on_pess, std_area_on_pess = np.mean(area_onshore_pess), np.std(area_onshore_pess)
# SOLAR
mean_area_solar_opt, std_area_solar_opt = np.mean(area_solar_opt), np.std(area_solar_opt)
mean_area_solar_most, std_area_solar_most = np.mean(area_solar_most), np.std(area_solar_most)
mean_area_solar_pess, std_area_solar_pess = np.mean(area_solar_pess), np.std(area_solar_pess)
# HYDRO
mean_area_hyd_opt, std_area_hyd_opt = np.mean(area_hydro_opt), np.std(area_hydro_opt)
mean_area_hyd_most, std_area_hyd_most = np.mean(area_hydro_most), np.std(area_hydro_most)
mean_area_hyd_pess, std_area_hyd_pess = np.mean(area_hydro_pess), np.std(area_hydro_pess)


# Function to calculate the % of Akershus area as benchmark

akershus_area = 5894 # km2

def km2_to_percentage(x):
    return (x / akershus_area) * 100

# OFFSHORE spatial requirements

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(area_offshore_opt, bins=bins, alpha=0.7, color="skyblue", hatch="...", label=f"Optimistic")
plt.axvline(x=mean_area_off_opt, color="black", linestyle="solid", label=f"Offshore Area μ ± 1σ = {mean_area_off_opt:,.0f} ± {std_area_off_opt:,.0f} km²")
plt.hist(area_offshore_most, bins=bins, alpha=0.7, color="skyblue", hatch="///", label=f"Most Likely")
plt.axvline(x=mean_area_off_most, color="black", linestyle="dashed", label=f"Offshore Area μ ± 1σ = {mean_area_off_most:,.0f} ± {std_area_off_most:,.0f} km² ")
plt.hist(area_offshore_pess, bins=bins, alpha=0.7, color="skyblue", hatch="---", label=f"Pessimistic")
plt.axvline(x=mean_area_off_pess, color="black", linestyle="dashdot", label=f"Offshore Area μ ± 1σ = {mean_area_off_pess:,.0f} ± {std_area_off_pess:,.0f} km² ")
plt.suptitle("Offshore Wind Area Requirements for Different Scenarios", fontsize=16, y=0.96)
plt.title("Benchmark: Akershus County Municipality (5894 km²)", fontsize=12)
plt.xlabel("Sea Area Required (km²)")
plt.ylabel("Probability Density")
plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5))
y_top = plt.ylim()[1]   # Placing text near mean
y_offset = 0.05 * y_top
plt.text(mean_area_off_opt * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_off_opt):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")
plt.text(mean_area_off_most * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_off_most):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")
plt.text(mean_area_off_pess * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_off_pess):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")


# ONSHORE spatial requirements
plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(area_onshore_opt, bins=bins, alpha=0.7, color="forestgreen", hatch= "...", label=f"Optimistic")
plt.axvline(x=mean_area_on_opt, color="black", linestyle="solid", label=f"Onshore Area μ ± 1σ = {mean_area_on_opt:,.0f} ± {std_area_on_opt:,.0f} km² ")
plt.hist(area_onshore_most, bins=bins, alpha=0.7, color="forestgreen", hatch= "///", label=f"Most Likely")
plt.axvline(x=mean_area_on_most, color="black", linestyle="dashed", label=f"Onshore Area μ ± 1σ = {mean_area_on_most:,.0f} ± {std_area_on_most:,.0f} km² ")
plt.hist(area_onshore_pess, bins=bins, alpha=0.7, color="forestgreen", hatch="---", label=f"Pessimistic")
plt.axvline(x=mean_area_on_pess, color="black", linestyle="dashdot", label=f"Onshore Area μ ± 1σ = {mean_area_on_pess:,.0f} ± {std_area_on_pess:,.0f} km² ")
plt.suptitle("Onshore Wind Area Requirements for Different Scenarios", fontsize=16, y=0.96)
plt.title("Benchmark: Akershus County Municipality (5894 km²)", fontsize=12)
plt.xlabel("Land Area Required (km²)")
plt.ylabel("Probability Density")
plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5))
y_top = plt.ylim()[1]
y_offset = 0.05 * y_top
plt.text(mean_area_on_opt * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_on_opt):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")
plt.text(mean_area_on_most * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_on_most):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")
plt.text(mean_area_on_pess * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_on_pess):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")



# SOLAR spatial requirements
plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(area_solar_opt, bins=bins, alpha=0.7, color="gold", hatch="...", label=f"Optimistic")
plt.axvline(x=mean_area_solar_opt, color="black", linestyle="solid", label=f"Solar Area μ ± 1σ = {mean_area_solar_opt:,.0f} ± {std_area_solar_opt:,.0f} km² ")
plt.hist(area_solar_most, bins=bins, alpha=0.7, color="gold",hatch="///", label=f"Most Likely")
plt.axvline(x=mean_area_solar_most, color="black", linestyle="dashed", label=f"Solar Area μ ± 1σ = {mean_area_solar_most:,.0f} ± {std_area_solar_most:,.0f} km² ")
plt.hist(area_solar_pess, bins=bins, alpha=0.7, color="gold", hatch="---",label=f"Pessimistic")
plt.axvline(x=mean_area_solar_pess, color="black", linestyle="dashdot", label=f"Solar Area μ ± 1σ = {mean_area_solar_pess:,.0f} ± {std_area_solar_pess:,.0f} km² ")
plt.suptitle("Solar PV Area Requirements for Different Scenarios", fontsize=16, y=0.96)
plt.title("Benchmark: Akershus County Municipality (5894 km²)", fontsize=12)
plt.xlabel("Land Area Required (km²)")
plt.ylabel("Probability Density")
plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5))
y_top = plt.ylim()[1]
y_offset = 0.05 * y_top
plt.text(mean_area_solar_opt * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_solar_opt):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")
plt.text(mean_area_solar_most * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_solar_most):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")
plt.text(mean_area_solar_pess * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_solar_pess):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")


# Plot for HYDRO
plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(area_hydro_opt, bins=bins, alpha=0.7, color="indianred", hatch="...",label=f"Optimistic")
plt.axvline(x=mean_area_hyd_opt, color="black", linestyle="solid", label=f"Hydro Area μ ± 1σ = {mean_area_hyd_opt:,.0f} ± {std_area_hyd_opt:,.0f} km² ")
plt.hist(area_hydro_most, bins=bins, alpha=0.7, color="indianred", hatch="///",label=f"Most Likely")
plt.axvline(x=mean_area_hyd_most, color="black", linestyle="dashed", label=f"Hydro Area μ ± 1σ = {mean_area_hyd_most:,.0f} ± {std_area_hyd_most:,.0f} km² ")
plt.hist(area_hydro_pess, bins=bins, alpha=0.7, color="indianred", hatch="---",label=f"Pessimistic")
plt.axvline(x=mean_area_hyd_pess, color="black", linestyle="dashdot", label=f"Hydro Area μ ± 1σ = {mean_area_hyd_pess:,.0f} ± {std_area_hyd_pess:,.0f} km² ")
plt.suptitle("Hydro Power Area Requirements for Different Scenarios (5.6 MW / km²)", fontsize=16, y=0.96)
plt.title("Benchmark: Akershus County Municipality (5894 km²)", fontsize=12)
plt.xlabel("Land Area Required (km²)")
plt.ylabel("Number of Observations")
plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5))
y_top = plt.ylim()[1]
y_offset = 0.05 * y_top
plt.text(mean_area_hyd_opt * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_hyd_opt):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")
plt.text(mean_area_hyd_most * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_hyd_most):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")
plt.text(mean_area_hyd_pess * 1.05, y_top - y_offset, f"{km2_to_percentage(mean_area_hyd_pess):.1f}% \n of Akershus",
        ha="left", va="top", fontsize=10, color="black")
plt.show()



# CO2 emissions from generating required electricity

# 1. Number of Modules needed to meet installed Capacity (Installed Capacity (MW) / (MW/module) )

# Module installed Capacity (MW / module). 1 module MW
Offshore_module = 8.6 # Referenced
Onshore_module = 4.2 # Referenced
Solar_module = 0.00059 # Calculated from reference
Hydro_module = 920 # Based on average MW of 4 largest hydropower plants in Norway

# Number of Offshore Turbines. Index 0 is offshore MW range in Monte Carlo
offshore_turbines_opt = MW_results_opt[:,0] / Offshore_module
offshore_turbines_most = MW_results_most[:,0]/ Offshore_module
offshore_turbines_pess = MW_results_pess[:,0] / Offshore_module

# Number of Onshore Turbines. Index 2 is onshore MW range
onshore_turbines_opt = MW_results_opt[:,2] / Onshore_module
onshore_turbines_most = MW_results_most[:,2]/ Onshore_module
onshore_turbines_pess = MW_results_pess[:,2] / Onshore_module

# Number of Solar panels. Index 4 is solar MW range
solar_cells_opt = MW_results_opt[:,4] / Solar_module
solar_cells_most =MW_results_most[:,4] / Solar_module
solar_cells_pess = MW_results_pess[:,4]/ Solar_module


# Number of Hydropower plants. Index 6 is hydro MW range
hydro_plants_opt = MW_results_opt[:,6]/ Hydro_module
hydro_plants_most = MW_results_most[:,6]/ Hydro_module
hydro_plants_pess =MW_results_pess[:,6] / Hydro_module

# 2. CO2 emissions per module
# Module Life Cycle emission factor. Reference: NREL
EF_off = 13
EF_on = 13
EF_sol = 43
EF_hyd = 21


# Calculated as 1 module MW * CF (range) * hrs/yr * EF * 1000 (from MWh to kWh)

# Offshore CO2 emissions from 1 module. Index 1 is offshore CF range in Monte Carlo
CO2_off_opt = Offshore_module * MW_results_opt[:,1] * 8760 *  EF_off *1000
CO2_off_most = Offshore_module * MW_results_most[:,1] * 8760 * EF_off*1000
CO2_off_pess = Offshore_module * MW_results_pess[:,1] * 8760 * EF_off*1000

# Onshore CO2 emissions from 1 module. Index 3 is onshore CF range
CO2_on_opt = Onshore_module * MW_results_opt[:,3] * 8760 *  EF_on *1000
CO2_on_most = Onshore_module * MW_results_most[:,3] * 8760 *  EF_on*1000
CO2_on_pess = Onshore_module * MW_results_pess[:,3] * 8760 *  EF_on*1000

# Solar CO2 emissions from 1 panel. Index 5 is solar CF range
CO2_sol_opt = Solar_module * MW_results_opt[:,5] * 8760 *  EF_sol*1000
CO2_sol_most = Solar_module * MW_results_most[:,5] * 8760 *  EF_sol*1000
CO2_sol_pess = Solar_module * MW_results_pess[:,5] * 8760 *  EF_sol*1000

# Hydro CO2 emissions from 1 plant. Index 7 is hydro CF range
CO2_hyd_opt = Hydro_module * MW_results_opt[:,7] * 8760  * EF_hyd*1000
CO2_hyd_most = Hydro_module * MW_results_most[:,7] * 8760  * EF_hyd*1000
CO2_hyd_pess = Hydro_module * MW_results_pess[:,7] * 8760 *  EF_hyd*1000

# 3. Total CO2 for entire installation is calculated by total number of modules * CO2 per module

# OFFSHORE Number of modules

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(offshore_turbines_opt, bins=bins, alpha=0.7, color="skyblue", hatch="...",label=f"Optimistic")
plt.axvline(x=np.mean(offshore_turbines_opt), color="black", linestyle="solid", label=f"Nr. of Offshore turbines μ ± 1σ = {np.mean(offshore_turbines_opt):,.0f} ± {np.std(offshore_turbines_opt):,.0f}")
plt.hist(offshore_turbines_most, bins=bins, alpha=0.7, color="skyblue", hatch="///",label=f"Most Likely")
plt.axvline(x=np.mean(offshore_turbines_most), color="black", linestyle="dashed", label=f"Nr. of Offshore turbines μ ± 1σ = {np.mean(offshore_turbines_most):,.0f} ± {np.std(offshore_turbines_most):,.0f}")
plt.hist(offshore_turbines_pess, bins=bins, alpha=0.7, color="skyblue", hatch="---",label=f"Pessimistic")
plt.axvline(x=np.mean(offshore_turbines_pess), color="black", linestyle="dashdot", label=f"Nr. of Offshore turbines μ ± 1σ = {np.mean(offshore_turbines_pess):,.0f} ± {np.std(offshore_turbines_pess):,.0f}")
plt.title("Offshore Wind Turbine Requirements for Different Scenarios \n(Each turbine = 8.6 MW Installed Capacity)", fontsize=14)
plt.xlabel("Number of Offshore Turbines", fontsize=12)
plt.ylabel("Number of Observations",fontsize=12)
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5))
plt.show()

# CO2 emissions OFFSHORE (number of modules * CO2 per module) / 1e12 (from grams to megatonnes)

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(offshore_turbines_opt * CO2_off_opt/1e12, bins=bins, alpha=0.7, color="skyblue", hatch="...",label=f"Optimistic")
plt.axvline(x=np.mean(offshore_turbines_opt * CO2_off_opt/1e12), color="black", linestyle="solid", label=f"Mt CO2e per year μ ± 1σ = {np.mean(offshore_turbines_opt * CO2_off_opt / 1e12):,.2f} ± {np.std(offshore_turbines_opt * CO2_off_opt / 1e12):,.2f} Mt")
plt.hist(offshore_turbines_most * CO2_off_most/1e12, bins=bins, alpha=0.7, color="skyblue", hatch="///",label=f"Most Likely")
plt.axvline(x=np.mean(offshore_turbines_most * CO2_off_most/1e12), color="black", linestyle="dashed", label=f"Mt CO2e per year μ ± 1σ = {np.mean(offshore_turbines_most * CO2_off_most / 1e12):,.2f} ± {np.std(offshore_turbines_most*CO2_off_most/ 1e12):,.2f} Mt")
plt.hist(offshore_turbines_pess * CO2_off_pess/1e12, bins=bins, alpha=0.7, color="skyblue", hatch="---",label=f"Pessimistic")
plt.axvline(x=np.mean(offshore_turbines_pess * CO2_off_pess/1e12), color="black", linestyle="dashdot", label=f"Mt CO2e per year μ ± 1σ = {np.mean(offshore_turbines_pess*CO2_off_pess / 1e12):,.2f} ± {np.std(offshore_turbines_pess*CO2_off_pess/ 1e12):,.2f} Mt")
plt.title("Distribution of Annual CO₂eq Emissions from Offshore Electricity \n(Optimistic, Most Likely & Pessimistic Scenarios)",fontsize=14)
plt.xlabel("Mt CO₂e per Year",fontsize=12)
plt.ylabel("Number of Observations",fontsize=12)
plt.legend(title = "Life Cycle emission factor: 13 g CO2-eq per kWh", loc = "center right", bbox_to_anchor=(1, 0.5))
plt.show()


# ONSHORE Number of modules

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(onshore_turbines_opt, bins=bins, alpha=0.7, color="forestgreen", hatch="...",label=f"Optimistic")
plt.axvline(x=np.mean(onshore_turbines_opt), color="black", linestyle="solid", label=f"Nr. of Onshore turbines μ ± 1σ = {np.mean(onshore_turbines_opt):,.0f} ± {np.std(onshore_turbines_opt):,.0f}")
plt.hist(onshore_turbines_most, bins=bins, alpha=0.7, color="forestgreen", hatch="///",label=f"Most Likely")
plt.axvline(x=np.mean(onshore_turbines_most), color="black", linestyle="dashed", label=f"Nr. of Onshore turbines μ ± 1σ = {np.mean(onshore_turbines_most):,.0f} ± {np.std(onshore_turbines_most):,.0f}")
plt.hist(onshore_turbines_pess, bins=bins, alpha=0.7, color="forestgreen", hatch="---",label=f"Pessimistic")
plt.axvline(x=np.mean(onshore_turbines_pess), color="black", linestyle="dashdot", label=f"Nr. of Onshore turbines μ ± 1σ = {np.mean(onshore_turbines_pess):,.0f} ± {np.std(onshore_turbines_pess):,.0f}")
plt.title("Onshore Wind Turbine Requirements for Different Scenarios \n(Each turbine = 4.2 MW Installed Capacity)", fontsize=14)
plt.xlabel("Number of Onshore Turbines", fontsize=12)
plt.ylabel("Number of Observations",fontsize=12)
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5))
plt.show()

# CO2 emissions ONSHORE (number of modules * CO2 per module) / 1e12 (from grams to megatonnes)

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(onshore_turbines_opt * CO2_on_opt/1e12, bins=bins, alpha=0.7, color="forestgreen", hatch="...",label=f"Optimistic")
plt.axvline(x=np.mean(onshore_turbines_opt * CO2_on_opt/1e12), color="black", linestyle="solid", label=f"Mt CO2e per year μ ± 1σ = {np.mean(onshore_turbines_opt * CO2_on_opt / 1e12):,.2f} ± {np.std(onshore_turbines_opt * CO2_on_opt / 1e12):,.2f} Mt")
plt.hist(onshore_turbines_most * CO2_on_most/1e12, bins=bins, alpha=0.7, color="forestgreen", hatch="///",label=f"Most Likely")
plt.axvline(x=np.mean(onshore_turbines_most * CO2_on_most/1e12), color="black", linestyle="dashed", label=f"Mt CO2e per year μ ± 1σ = {np.mean(onshore_turbines_most * CO2_on_most / 1e12):,.2f} ± {np.std(onshore_turbines_most*CO2_on_most/ 1e12):,.2f} Mt")
plt.hist(onshore_turbines_pess * CO2_on_pess/1e12, bins=bins, alpha=0.7, color="forestgreen", hatch="---",label=f"Pessimistic")
plt.axvline(x=np.mean(onshore_turbines_pess * CO2_on_pess/1e12), color="black", linestyle="dashdot", label=f"Mt CO2e per year μ ± 1σ = {np.mean(onshore_turbines_pess*CO2_on_pess / 1e12):,.2f} ± {np.std(onshore_turbines_pess*CO2_on_pess/ 1e12):,.2f} Mt")
plt.title("Distribution of Annual CO₂eq Emissions from Onshore Electricity \n(Optimistic, Most Likely & Pessimistic Scenarios)", fontsize=14)
plt.xlabel("Mt CO₂e per Year",fontsize=12)
plt.ylabel("Number of Observations",fontsize=12)
plt.legend(title = "Life Cycle emission factor: 13 g CO2-eq per kWh",loc = "center right", bbox_to_anchor=(1, 0.5))
plt.show()

# SOLAR Number of panels

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(solar_cells_opt, bins=bins, alpha=0.7, color="gold", hatch="...",label=f"Optimistic")
plt.axvline(x=np.mean(solar_cells_opt), color="black", linestyle="solid", label=f"Nr. of Solar PV modules μ ± 1σ = {np.mean(solar_cells_opt):,.0f} ± {np.std(solar_cells_opt):,.0f}")
plt.hist(solar_cells_most, bins=bins, alpha=0.7, color="gold", hatch="///",label=f"Most Likely")
plt.axvline(x=np.mean(solar_cells_most), color="black", linestyle="dashed", label=f"Nr. of Solar PV modules μ ± 1σ = {np.mean(solar_cells_most):,.0f} ± {np.std(solar_cells_most):,.0f}")
plt.hist(solar_cells_pess, bins=bins, alpha=0.7, color="gold", hatch="---",label=f"Pessimistic")
plt.axvline(x=np.mean(solar_cells_pess), color="black", linestyle="dashdot", label=f"Nr. of Solar PV modules μ ± 1σ = {np.mean(solar_cells_pess):,.0f} ± {np.std(solar_cells_pess):,.0f}")
plt.title("Solar PV Module Requirements for Different Scenarios \n(Each module = 0.00059 MW Installed Capacity)", fontsize=14)
plt.xlabel("Number of Solar Modules", fontsize=12)
plt.ylabel("Number of Observations",fontsize=12)
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5))
plt.show()

# CO2 emissions SOLAR (number of panels * CO2 per panel) / 1e12 (from grams to megatonnes)

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(solar_cells_opt * CO2_sol_opt/1e12, bins=bins, alpha=0.7, color="gold", hatch="...",label=f"Optimistic")
plt.axvline(x=np.mean(solar_cells_opt * CO2_sol_opt/1e12), color="black", linestyle="solid", label=f"Mt CO2e per year μ ± 1σ = {np.mean(solar_cells_opt * CO2_sol_opt/1e12):,.2f} ± {np.std(solar_cells_opt * CO2_sol_opt/1e12):,.2f} Mt")
plt.hist(solar_cells_most * CO2_sol_most/1e12, bins=bins, alpha=0.7, color="gold", hatch="///",label=f"Most Likely")
plt.axvline(x=np.mean(solar_cells_most * CO2_sol_most/1e12), color="black", linestyle="dashed", label=f"Mt CO2e per year μ ± 1σ = {np.mean(solar_cells_most * CO2_sol_most/1e12):,.2f} ± {np.std(solar_cells_most * CO2_sol_most/1e12):,.2f} Mt")
plt.hist(solar_cells_pess * CO2_sol_pess/1e12, bins=bins, alpha=0.7, color="gold", hatch="---",label=f"Pessimistic")
plt.axvline(x=np.mean(solar_cells_pess * CO2_sol_pess/1e12), color="black", linestyle="dashdot", label=f"Mt CO2e per year μ ± 1σ = {np.mean(solar_cells_pess * CO2_sol_pess/1e12):,.2f} ± {np.std(solar_cells_pess * CO2_sol_pess/1e12):,.2f} Mt")
plt.title("Distribution of Annual CO₂eq Emissions from Solar PV Electricity \n(Optimistic, Most Likely & Pessimistic Scenarios)", fontsize=14)
plt.xlabel("Mt CO₂e per Year",fontsize=12)
plt.ylabel("Number of Observations",fontsize=12)
plt.legend(title = "Life Cycle Emission factor: 43 g CO2-eq per kWh", loc = "center right", bbox_to_anchor=(1, 0.5))
plt.show()


# TESTING FOR SAME CO2 RESULT: CALCULATED WITH SOLAR EF AND TOTAL ELECTRICITY REQUIREMENT - CONFIRMED
#plt.figure(figsize=(10, 6))
#plt.xscale("log")
#plt.hist(energy_optimistic * 1e9 * 43/1e12, bins=bins, alpha=0.7, color="gold", hatch="...",label=f"Optimistic")
#plt.axvline(x=np.mean(energy_optimistic * 1e9 * 43/1e12), color="black", linestyle="solid", label=f"Mt CO2e per year μ ± 1σ = {np.mean(energy_optimistic * 1e9 * 43/1e12):,.2f} ± {np.std(energy_optimistic * 1e9 * 43/1e12):,.2f} Mt")
#plt.hist(energy_most_likely * 1e9 * 43 /1e12, bins=bins, alpha=0.7, color="gold", hatch="///",label=f"Most Likely")
#plt.axvline(x=np.mean(energy_most_likely * 1e9 * 43/1e12), color="black", linestyle="dashed", label=f"Mt CO2e per year μ ± 1σ = {np.mean(energy_most_likely * 1e9 * 43/1e12):,.2f} ± {np.std(energy_most_likely * 1e9 * 43/1e12):,.2f} Mt")
#plt.hist(energy_pessimistic * 1e9 * 43/1e12, bins=bins, alpha=0.7, color="gold", hatch="---",label=f"Pessimistic")
#plt.axvline(x=np.mean(energy_pessimistic * 1e9 * 43/1e12), color="black", linestyle="dashdot", label=f"Mt CO2e per year μ ± 1σ = {np.mean(energy_pessimistic * 1e9 * 43/1e12):,.2f} ± {np.std(energy_pessimistic * 1e9 * 43/1e12):,.2f} Mt")
#plt.title("Annual Solar Power CO2eq Emissions for Different Scenarios")
#plt.xlabel("Mt CO2e per year")
#plt.ylabel("Number of Observations")
#plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5))
#plt.show()

# HYDRO Number of plants

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(hydro_plants_opt, bins=bins, alpha=0.7, color="indianred", hatch="...",label=f"Optimistic")
plt.axvline(x=np.mean(hydro_plants_opt), color="black", linestyle="solid", label=f"Nr. of Hydro Power Facilities μ ± 1σ = {np.mean(hydro_plants_opt):,.0f} ± {np.std(hydro_plants_opt):,.0f}")
plt.hist(hydro_plants_most, bins=bins, alpha=0.7, color="indianred", hatch="///",label=f"Most Likely")
plt.axvline(x=np.mean(hydro_plants_most), color="black", linestyle="dashed", label=f"Nr. of Hydro Power Facilities μ ± 1σ = {np.mean(hydro_plants_most):,.0f} ± {np.std(hydro_plants_most):,.0f}")
plt.hist(hydro_plants_pess, bins=bins, alpha=0.7, color="indianred", hatch="---",label=f"Pessimistic")
plt.axvline(x=np.mean(hydro_plants_pess), color="black", linestyle="dashdot", label=f"Nr. of Hydro Power Facilities μ ± 1σ = {np.mean(hydro_plants_pess):,.0f} ± {np.std(hydro_plants_pess):,.0f}")
plt.title("Hydropower Facility Requirements for Different Scenarios \n(Each facility = 920 MW Installed Capacity)", fontsize=14)
plt.xlabel("Number of Hydro Power Facilities", fontsize=12)
plt.ylabel("Number of Observations",fontsize=12)
plt.legend(loc = "center right", bbox_to_anchor=(1, 0.5))
plt.show()

#CO2 emissions HYDRO (number of plants * CO2 per plant) / 1e12 (from grams to megatonnes)

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.hist(hydro_plants_opt * CO2_hyd_opt/1e12, bins=bins, alpha=0.7, color="indianred", hatch="...",label=f"Optimistic")
plt.axvline(x=np.mean(hydro_plants_opt * CO2_hyd_opt/1e12), color="black", linestyle="solid", label=f"Mt CO2e per year μ ± 1σ = {np.mean(hydro_plants_opt * CO2_hyd_opt/1e12):,.2f} ± {np.std(hydro_plants_opt * CO2_hyd_opt/1e12):,.2f} Mt")
plt.hist(hydro_plants_most * CO2_hyd_most/1e12, bins=bins, alpha=0.7, color="indianred", hatch="///",label=f"Most Likely")
plt.axvline(x=np.mean(hydro_plants_most * CO2_hyd_most/1e12), color="black", linestyle="dashed", label=f"Mt CO2e per year μ ± 1σ = {np.mean(hydro_plants_most * CO2_hyd_most/1e12):,.2f} ± {np.std(hydro_plants_most * CO2_hyd_most/1e12):,.2f} Mt")
plt.hist(hydro_plants_pess * CO2_hyd_pess/1e12, bins=bins, alpha=0.7, color="indianred", hatch="---",label=f"Pessimistic")
plt.axvline(x=np.mean(hydro_plants_pess * CO2_hyd_pess/1e12), color="black", linestyle="dashdot", label=f"Mt CO2e per year μ ± 1σ = {np.mean(hydro_plants_pess * CO2_hyd_pess/1e12):,.2f} ± {np.std(hydro_plants_pess * CO2_hyd_pess/1e12):,.2f} Mt")
plt.title("Distribution of Annual CO₂eq Emissions from Hydro Power Electricity \n(Optimistic, Most Likely & Pessimistic Scenarios)", fontsize=14)
plt.xlabel("Mt CO₂e per Year", fontsize=12)
plt.ylabel("Number of Observations",fontsize=12)
plt.legend(title = "Life Cycle Emission factor: 21 g CO2-eq per kWh", loc = "center right", bbox_to_anchor=(1, 0.5))
plt.show()