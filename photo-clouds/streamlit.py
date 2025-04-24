import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from astropy.timeseries import LombScargle

from update import neptune_wind
from light_curve import *

# shamelessly stolen from https://github.com/earlbellinger/night-time-sine/
plt.rcParams.update({'axes.linewidth' : 1.5,
                     'ytick.major.width' : 1.5,
                     'ytick.minor.width' : 1.5,
                     'xtick.major.width' : 1.5,
                     'xtick.minor.width' : 1.5,
                     'xtick.labelsize': 12, 
                     'ytick.labelsize': 12,
                     'axes.labelsize': 18,
                     'axes.labelpad' : 5,
                     'axes.titlesize' : 22,
                     'axes.titlepad' : 10,
                     'font.family': 'Serif'
                    })

def plot_wind_profile(wind_profile, latitudes, title):
    
    fig, ax = plt.subplots()
    ax.plot(wind_profile, latitudes, label=f"{title} Zonal Wind Profile")
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Latitude [°]")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

def plot_light_curve(times, lightcurve, title):
    
    fig, ax = plt.subplots()
    ax.plot(times, normalize(lightcurve), label=f"{title} Light Curve")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Normalized Flux")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

def plot_periodogram(frequency, power, false_alarm, title):
    
    fig, ax = plt.subplots()
    ax.plot(frequency, power, label=f"{title} Lomb-Scargle Periodogram")
    ax.set_xlabel("Frequency [1/days]")
    ax.set_ylabel("Power")
    ax.axhline(false_alarm, c='black', linestyle='dashed', label='False Alarm Probability Line')
    ax.set_xlim(0.1, 3)
    ax.legend()
    ax.grid()
    st.pyplot(fig)

# Streamlit page

st.title("Model Light Curve of a Planetary Albedo with Transient Features")
st.sidebar.title("Controls")

wind_model = st.sidebar.selectbox('Select a latitudinal wind speed profile.', ['Uranus', 'Neptune', 'Custom'])
st.sidebar.write(f"{wind_model} latitudinal wind speed selected.")

if wind_model == 'Neptune':
    plot_wind_profile(neptune_wind(np.linspace(-90, 90, 100)), np.linspace(-90, 90, 100), wind_model)

if wind_model == 'Custom':
    # Get the custom formula from the user as a string
    custom_formula = st.sidebar.text_area("Enter a custom wind formula as a function of latitude", "e.g., 'sin(x)'")

    # Define a safe dictionary with numpy functions
    safe_dict = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'log': np.log,
        'exp': np.exp,
        'sqrt': np.sqrt,
        'pi': np.pi,
        'x': np.linspace(-10, 10, 100)  # Example range for x (or you can use a single value, e.g., x = 5)
    }

    if custom_formula.strip() == "" or custom_formula == "e.g., 'sin(x)'":
        st.sidebar.error("Please enter a valid custom formula.")
    else:
        try:
            # Use eval to calculate the wind profile formula, safely evaluating it with the numpy functions
            wind_profile_value = eval(custom_formula, {"__builtins__": None}, safe_dict)

            # Display the result
            st.write(f"Custom Formula: {custom_formula}")

            plot_wind_profile(wind_profile_value, np.linspace(-90, 90, 100), wind_model)

        except Exception as e:
            st.sidebar.error(f"Error evaluating formula: {e}")
#clouds
if 'clouds' not in st.session_state:
    st.session_state.clouds = []

# Cloud options for the dropdown (including an empty first option)
cloud_options = [""] + [f"Cloud {i + 1}" for i in range(len(st.session_state.clouds))]
cloud_options.append("+ Add Cloud")  # Option to add a new cloud

# Sidebar: Select a cloud or add a new one, with a blank option at the top
selected_cloud = st.sidebar.selectbox("Select a cloud or add a new one:", cloud_options)

# Handle adding a new cloud
if selected_cloud == "+ Add Cloud":
    # If the user selects "+ Add Cloud", add a new cloud to the list in session_state
    new_cloud_id = len(st.session_state.clouds) + 1
    cloud_name = f"Cloud {new_cloud_id}"
    st.sidebar.write(f"Adding {cloud_name} to the list!")

    # Initialize input fields in session_state if not already initialized
    if 'lat' not in st.session_state:
        st.session_state.lat = None
    if 'long' not in st.session_state:
        st.session_state.long = None
    if 'flux' not in st.session_state:
        st.session_state.flux = None
    if 'a' not in st.session_state:
        st.session_state.a = None
    if 'b' not in st.session_state:
        st.session_state.b = None

    # Inputs for latitude, longitude, and altitude (values persist in session_state)
    st.session_state.lat = st.sidebar.number_input(f"Latitude for {cloud_name}", min_value=0.0, max_value=100.0, value=st.session_state.lat)
    st.session_state.long = st.sidebar.number_input(f"Initial longitude for {cloud_name}", min_value=0.0, max_value=100.0, value=st.session_state.long)
    st.session_state.flux = st.sidebar.number_input(f"Net flux for {cloud_name}", min_value=0.0, max_value=1000.0, value=st.session_state.flux)
    st.session_state.a = st.sidebar.number_input(f"Latitudinal extent for {cloud_name}", min_value=0.0, max_value=1000.0, value=st.session_state.a)
    st.session_state.b = st.sidebar.number_input(f"Longitudinal extent for {cloud_name}", min_value=0.0, max_value=1000.0, value=st.session_state.b)

    # Add the new cloud if all inputs are filled (non-None)
    if all(v is not None for v in [st.session_state.lat, st.session_state.long, st.session_state.flux, st.session_state.a, st.session_state.b]):
        if st.sidebar.button("Save Cloud"):
            st.session_state.clouds.append({
                'name': cloud_name,
                'lat': st.session_state.lat,
                'long': st.session_state.long,
                'flux': st.session_state.flux,
                'a': st.session_state.a,
                'b': st.session_state.b
            })
            st.sidebar.write(f"{cloud_name} has been added!")

    else:
        st.sidebar.write("Please fill in all fields before saving the cloud.")

# Optional: Display the selected cloud details if it's not blank
if selected_cloud and selected_cloud != "+ Add Cloud":
    try:
        # Extract cloud index from "Cloud X" (e.g., "Cloud 1" -> index 0)
        cloud_index = int(selected_cloud.split()[1]) - 1  # Extract cloud index from "Cloud X"
        cloud = st.session_state.clouds[cloud_index]
        
        # Display the cloud's input fields (latitude, longitude, altitude, etc.)
        st.sidebar.write(f"Modify details for {cloud['name']}:")
        
        cloud['lat'] = st.sidebar.number_input(f"Latitude for {cloud['name']}", min_value=0.0, max_value=100.0, value=cloud['lat'])
        cloud['long'] = st.sidebar.number_input(f"Longitude for {cloud['name']}", min_value=0.0, max_value=100.0, value=cloud['long'])
        cloud['flux'] = st.sidebar.number_input(f"Altitude for {cloud['name']}", min_value=0.0, max_value=1000.0, value=cloud['flux'])
        cloud['a'] = st.sidebar.number_input(f"Altitude for {cloud['name']}", min_value=0.0, max_value=1000.0, value=cloud['a'])
        cloud['b'] = st.sidebar.number_input(f"Altitude for {cloud['name']}", min_value=0.0, max_value=1000.0, value=cloud['b'])

        if st.sidebar.button("Save Cloud"):
            st.sidebar.write(f"{cloud['name']} has been updated!")
            
    except IndexError:
        st.sidebar.error("Error: Invalid cloud selection.")
else:
    st.sidebar.write("Please select a valid cloud.")

# light curve
if st.sidebar.button("Generate Light Curve"):
    latitudes = [cloud['lat'] for cloud in st.session_state.clouds]
    longitudes = [cloud['long'] for cloud in st.session_state.clouds]
    fluxes = [cloud['flux'] for cloud in st.session_state.clouds]
    a_vals = [cloud['a'] for cloud in st.session_state.clouds]
    b_vals = [cloud['b'] for cloud in st.session_state.clouds]
    
    times = np.linspace(0, 30, 100)

    lightcurve = make_lightcurve(times, fluxes, latitudes, longitudes=longitudes, a_arr=a_vals, b_arr=b_vals)
    st.session_state.lightcurve = lightcurve
    plot_light_curve(times, lightcurve, wind_model)

#periodogram
if 'lightcurve' in st.session_state and st.session_state.lightcurve is not None:
    if st.sidebar.button("Generate Periodogram"):
        lightcurve = st.session_state.lightcurve  # Get the light curve from session_state
        times = np.linspace(0, 30, 100)

        # Perform LombScargle periodogram analysis
        ls = LombScargle(times, lightcurve)
        frequency, power = ls.autopower(samples_per_peak=10)
        false_alarm = ls.false_alarm_level(0.01)
        plot_periodogram(frequency, power, false_alarm, wind_model)
else:
    st.sidebar.warning("Please generate the light curve first.")

        