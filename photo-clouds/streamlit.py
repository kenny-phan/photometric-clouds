import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from astropy.timeseries import LombScargle

from update import neptune_wind
from light_curve import *
from animate import run_animation

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

def plot_photo_clouds(times_obs, flux_obs, lc):
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, 
                        gridspec_kw={'height_ratios': [2, 0.7]})
    
    axs[0].plot(times_obs, normalize(flux_obs), label="Light Curve Data Input", color="xkcd:blue")
    axs[0].plot(times_obs, normalize(lc), label="Simulated Light Curve", color="xkcd:orange")
    axs[0].legend()
    axs[0].grid()
    axs[0].set_ylabel("Normalized Flux")
    
    axs[1].plot(times_obs, normalize(flux_obs) - normalize(lc), color='black')
    axs[1].tick_params(labeltop=False)
    axs[1].grid()
    axs[1].set_ylabel("Residual Flux")
    
    fig.supxlabel("Time [days]")  # Global x-axis label

    st.pyplot(fig)

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

def plot_periodogram(frequency, power, false_alarm, title, sim_freq=None, sim_pow=None, sim_fap=None):
    
    fig, ax = plt.subplots()
    ax.plot(frequency, power, label=f"{title} Lomb-Scargle Periodogram", color="xkcd:blue")
    ax.set_xlabel("Frequency [1/days]")
    ax.set_ylabel("Power")
    ax.axhline(false_alarm, c='black', linestyle='dashed', label='False Alarm Probability Line')

    if sim_freq is not None and sim_pow is not None and sim_fap is not None:
        ax.plot(sim_freq, sim_pow, label=f"Simulated Lomb-Scargle Periodogram", color="xkcd:orange")
        ax.axhline(sim_fap, c='grey', linestyle='dashed', label='Simulated False Alarm Probability Line')
    
    ax.set_xlim(0.1, 3)
    ax.legend()
    ax.grid()   
    st.pyplot(fig)


def wind_model_controls():
    """
    Generalized function to handle wind model selection and custom wind formula input.
    """
    st.sidebar.title("Controls")
    
    # Wind model selection
    wind_model = st.sidebar.selectbox('Select a latitudinal wind speed profile.', ['Neptune', 'Custom'])
    st.sidebar.write(f"{wind_model} latitudinal wind speed selected.")
    
    # Handling different wind models
    if wind_model == 'Neptune':
        plot_wind_profile(neptune_wind(np.linspace(-90, 90, 100)), np.linspace(-90, 90, 100), wind_model)
    
    elif wind_model == 'Custom':
        custom_formula = st.sidebar.text_area("Enter a custom wind formula as a function of latitude in units of meters per second", "e.g., 'sin(x)'")
        
        # Safe dictionary for eval()
        safe_dict = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'log': np.log,
            'exp': np.exp,
            'sqrt': np.sqrt,
            'pi': np.pi,
            'x': np.linspace(-10, 10, 100)  # Example range for x
        }
        
        if custom_formula.strip() == "" or custom_formula == "e.g., 'sin(x)'":
            st.sidebar.error("Please enter a valid custom formula.")
        else:
            try:
                wind_profile_value = eval(custom_formula, {"__builtins__": None}, safe_dict)
                st.write(f"Custom Formula: {custom_formula}")
                plot_wind_profile(wind_profile_value, np.linspace(-90, 90, 100), wind_model)
            except Exception as e:
                st.sidebar.error(f"Error evaluating formula: {e}")

    return wind_model

def file_process(uploaded_file, wind_model):
    if uploaded_file is not None:
        try:
            # Check if data already exists in session state
            if 'times' not in st.session_state or 'flux' not in st.session_state or st.session_state.get('wind_model') != wind_model:
                print(wind_model)
                # Load file content and run the photo weather model
                times, flux = load_lines(uploaded_file)
                
                # Run photo weather model and ensure each return value is checked properly
                sin, lc, lats_transformed, fit_offset, fit_amp, frequency_dt, power_dt, false_alarm_dt = run_photo_weather(times, flux, detrend_count=1, object_name=wind_model)
                
                # Ensure lats_transformed is not empty or None before processing
                if lats_transformed is not None and lats_transformed.size > 0:
                    # Store the results in session_state
                    st.session_state.times = times
                    st.session_state.flux = flux
                    st.session_state.sin = sin
                    st.session_state.lc = lc
                    st.session_state.lats_transformed = lats_transformed
                    st.session_state.fit_offset = fit_offset
                    st.session_state.fit_amp = fit_amp
                    st.session_state.frequency_dt = frequency_dt
                    st.session_state.power_dt = power_dt
                    st.session_state.false_alarm_dt = false_alarm_dt
                    st.session_state.wind_model = wind_model  # Store the selected wind model to avoid recalculating
                else:
                    st.error("Error: No latitudes were transformed. Please check the input file or model.")
                    return
                
            else:
                # Use the existing data in session state if available
                times = st.session_state.times
                flux = st.session_state.flux
                sin = st.session_state.sin
                lc = st.session_state.lc
                lats_transformed = st.session_state.lats_transformed
                fit_offset = st.session_state.fit_offset
                fit_amp = st.session_state.fit_amp
                frequency_dt = st.session_state.frequency_dt
                power_dt = st.session_state.power_dt
                false_alarm_dt = st.session_state.false_alarm_dt

            # Plot the photo clouds (Light curve)
            plot_photo_clouds(times, flux, lc)

            # Generate periodograms using the stored data
            ls = LombScargle(times, lc)
            frequency, power = ls.autopower(samples_per_peak=3)
            false_alarm = ls.false_alarm_level(0.0001)

            # Plot the periodogram (Live Update)
            plot_periodogram(frequency_dt, power_dt, false_alarm_dt, wind_model, sim_freq=frequency, sim_pow=power, sim_fap=false_alarm)
            
            # Now handle latitude parameter updates and auto-update light curve
            update_latitude_parameters(lats_transformed, times, flux, frequency_dt, power_dt, false_alarm_dt, wind_model)

        except Exception as e:
            st.error(f"Failed to load file: {e}")        
        
    # No need for a button here; the updates happen automatically now
    if st.sidebar.button("Animate from data!"):
        # Use stored animation data
        t = np.linspace(0, 100 * 3600, 100)
        ani = run_animation(fit_offset, lats_transformed, t)
        html_ani = ani.to_jshtml()
        components.html(html_ani, height=1000, width=1600)


def update_latitude_parameters(lats_transformed, times, data_flux, frequency_dt, power_dt, false_alarm_dt, wind_model):
    """
    Allow the user to update flux, a, and b for each latitude.
    This will auto-update the light curve whenever parameters are changed.
    """
    # Initialize clouds list in session_state if not exists
    if 'clouds' not in st.session_state:
        st.session_state.clouds = []

    # If latitudes are provided, create the parameter fields for each latitude
    if lats_transformed is not None and lats_transformed.size > 0:
        # Initialize clouds with latitudes and default values for flux, a, b
        for lat in lats_transformed:
            if not any(cloud['lat'] == lat for cloud in st.session_state.clouds):  # Avoid duplicate latitudes
                st.session_state.clouds.append({'lat': lat, 'flux': 100.0, 'a': 6.0, 'b': 2.0})

    # Now display fields to modify flux, a, and b for each latitude
    for i, cloud in enumerate(st.session_state.clouds):
        st.sidebar.subheader(f"Cloud {i + 1} Parameters (Latitude: {cloud['lat']}°)")

        # Use number_input widgets to allow the user to modify flux, a, and b
        new_flux = st.sidebar.slider(f"Flux for Cloud {i + 1}", min_value=-100.0, max_value=100.0, value=cloud['flux'], step=10.0)
        new_a = st.sidebar.slider(f"Latitudinal extent (a) for Cloud {i + 1}", min_value=0.0, max_value=50.0, value=cloud['a'], step=1.0)
        new_b = st.sidebar.slider(f"Longitudinal extent (b) for Cloud {i + 1}", min_value=0.0, max_value=50.0, value=cloud['b'], step=1.0)

        # Update the cloud parameters if any value has changed
        if cloud['flux'] != new_flux or cloud['a'] != new_a or cloud['b'] != new_b:
            # Update the cloud parameters in session_state
            st.session_state.clouds[i] = {'lat': cloud['lat'], 'flux': new_flux, 'a': new_a, 'b': new_b}
            # Set the flag to trigger auto update
            st.session_state.updated_clouds = True

    # If any values have been updated, recalculate the light curve and periodogram automatically
    if 'updated_clouds' in st.session_state and st.session_state.updated_clouds:
        st.session_state.updated_clouds = False  # Reset the flag after update
        generate_updated_light_curve(times, data_flux, frequency_dt, power_dt, false_alarm_dt, wind_model)


def generate_updated_light_curve(times, data_flux, frequency_dt, power_dt, false_alarm_dt, wind_model):
    """
    Generate and plot the light curve after updating the parameters automatically.
    """
    # Extract updated parameters from session_state
    latitudes = [cloud['lat'] for cloud in st.session_state.clouds]
    fluxes = [cloud['flux'] for cloud in st.session_state.clouds]
    a_vals = [cloud['a'] for cloud in st.session_state.clouds]
    b_vals = [cloud['b'] for cloud in st.session_state.clouds]

    # Generate the light curve with the updated parameters
    lightcurve = make_lightcurve(times, fluxes, latitudes, a_arr=a_vals, b_arr=b_vals)
    
    # Plot the updated light curve
    plot_photo_clouds(times, data_flux, lightcurve)

    ls = LombScargle(times, lightcurve)
    frequency, power = ls.autopower(samples_per_peak=3)
    false_alarm = ls.false_alarm_level(0.0001)

    # Plot the periodogram (Live Update)
    plot_periodogram(frequency_dt, power_dt, false_alarm_dt, wind_model, sim_freq=frequency, sim_pow=power, sim_fap=false_alarm)

    # Optionally show some confirmation message or update the state
    st.success("Light curve has been updated with new parameters!")

def sidebar_controls(wind_model):
    
    # Cloud handling
    if 'clouds' not in st.session_state:
        st.session_state.clouds = []

    # Cloud options for the dropdown (including an empty first option)
    cloud_options = [""] + [f"Cloud {i + 1}" for i in range(len(st.session_state.clouds))]
    cloud_options.append("+ Add Cloud")  # Option to add a new cloud

    # Sidebar: Select a cloud or add a new one
    selected_cloud = st.sidebar.selectbox("Select a cloud or add a new one:", cloud_options)
    
    if selected_cloud == "+ Add Cloud":
        new_cloud_id = len(st.session_state.clouds) + 1
        cloud_name = f"Cloud {new_cloud_id}"
        st.sidebar.write(f"Adding {cloud_name} to the list!")
        
        # Initialize session_state for the cloud's parameters
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
        
        # Input fields for new cloud
        st.session_state.lat = st.sidebar.number_input(f"Latitude for {cloud_name}", min_value=-90.0, max_value=90.0, value=st.session_state.lat)
        st.session_state.long = st.sidebar.number_input(f"Initial longitude for {cloud_name}", min_value=-90.0, max_value=90.0, value=st.session_state.long)
        st.session_state.flux = st.sidebar.number_input(f"Net flux for {cloud_name}", min_value=-1000.0, max_value=1000.0, value=st.session_state.flux)
        st.session_state.a = st.sidebar.number_input(f"Latitudinal extent for {cloud_name}", min_value=0.0, max_value=1000.0, value=st.session_state.a)
        st.session_state.b = st.sidebar.number_input(f"Longitudinal extent for {cloud_name}", min_value=0.0, max_value=1000.0, value=st.session_state.b)

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
    
    # Modify existing cloud details
    if selected_cloud and selected_cloud != "+ Add Cloud":
        try:
            cloud_index = int(selected_cloud.split()[1]) - 1
            cloud = st.session_state.clouds[cloud_index]
            st.sidebar.write(f"Modify details for {cloud['name']}:")
            cloud['lat'] = st.sidebar.number_input(f"Latitude for {cloud['name']}", min_value=-90.0, max_value=90.0, value=cloud['lat'])
            cloud['long'] = st.sidebar.number_input(f"Longitude for {cloud['name']}", min_value=-90.0, max_value=90.0, value=cloud['long'])
            cloud['flux'] = st.sidebar.number_input(f"Net flux for {cloud['name']}", min_value=-1000.0, max_value=1000.0, value=cloud['flux'])
            cloud['a'] = st.sidebar.number_input(f"Latitudinal extent for {cloud['name']}", min_value=0.0, max_value=1000.0, value=cloud['a'])
            cloud['b'] = st.sidebar.number_input(f"Longitudinal extent for {cloud['name']}", min_value=0.0, max_value=1000.0, value=cloud['b'])
            
            if st.sidebar.button("Save Cloud"):
                st.sidebar.write(f"{cloud['name']} has been updated!")
        except IndexError:
            st.sidebar.error("Error: Invalid cloud selection.")
    
    # Light Curve Generation
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
    
    # Periodogram Generation
    if 'lightcurve' in st.session_state and st.session_state.lightcurve is not None:
        if st.sidebar.button("Generate Periodogram"):
            lightcurve = st.session_state.lightcurve
            times = np.linspace(0, 30, 100)
            ls = LombScargle(times, lightcurve)
            frequency, power = ls.autopower(samples_per_peak=10)
            false_alarm = ls.false_alarm_level(0.05)
            plot_periodogram(frequency, power, false_alarm, wind_model)
    else:
        st.sidebar.warning("Please generate the light curve first.")
    
    # Animation Generation
    if st.sidebar.button("Animate!"):
        times = np.linspace(0, 30 * 3600 * 24, 100)
        latitudes = [cloud['lat'] for cloud in st.session_state.clouds]
        longitudes = [cloud['long'] for cloud in st.session_state.clouds]
        fluxes = [cloud['flux'] for cloud in st.session_state.clouds]
        a_vals = [cloud['a'] for cloud in st.session_state.clouds]
        b_vals = [cloud['b'] for cloud in st.session_state.clouds]
        
        ani = run_animation(longitudes, latitudes, times, a_arr=a_vals, b_arr=b_vals, fluxes=fluxes)
        html_ani = ani.to_jshtml()
        components.html(html_ani, height=1000, width=1600)

# Main
def main():
    st.title("Model Light Curve of a Planetary Albedo with Transient Features")
    wind_model = wind_model_controls()  # Call to wind model controls

    # Initialize session state variables if they don't exist yet
    if 'show_file_uploader' not in st.session_state:
        st.session_state.show_file_uploader = False  # Ensure the file uploader is hidden initially
    if 'sidebar_open' not in st.session_state:
        st.session_state.sidebar_open = False  # Initialize sidebar state as closed

    # Button to trigger file process (only visible if sidebar button wasn't pressed)
    file_button = st.button("I want to upload my own light curve.")

    if file_button:
        # Reset sidebar and any data/plots from the previous button click
        st.session_state.show_file_uploader = True
        st.session_state.sidebar_open = False  # Close the sidebar if it was opened
        st.session_state.lightcurve = None  # Reset light curve plot if needed

    # Show the file uploader only after the file_button is pressed
    if st.session_state.show_file_uploader:
        uploaded_file = st.file_uploader("Upload light curve as a .txt file", type=["txt"])

        # Process the file once it's uploaded
        if uploaded_file:
            file_process(uploaded_file, wind_model)  # Pass uploaded file and wind model

    # Button to open the sidebar for controls (only visible if file button wasn't pressed)
    sidebar_button = st.button("I want to generate a light curve from albedo parameters.")
    
    if sidebar_button:
        # Reset file uploader and clear any plots when sidebar button is clicked
        st.session_state.sidebar_open = True # Toggle sidebar visibility
        st.session_state.show_file_uploader = False  # Hide file uploader
        st.session_state.lightcurve = None  # Reset light curve plot if needed

    # Only show sidebar if the button was clicked
    if st.session_state.sidebar_open:
        sidebar_controls(wind_model)  # Call to sidebar controls for generating the light curve


if __name__ == "__main__":
    main()
