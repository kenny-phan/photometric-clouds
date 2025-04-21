import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from light_curve import *
from animate import *

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

def plot(times_obs, flux_obs):
    sin, lc = run_photo_weather(times_obs, flux_obs)
    
    fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True, 
                        gridspec_kw={'height_ratios': [2, 2, 0.7, 2, 0.7]})
    
    axs[0].plot(times_obs, normalize(flux_obs), label="data", color="xkcd:royal blue")
    axs[0].legend()

    axs[1].plot(times_obs, normalize(lc), label="peak frequencies", color="xkcd:green")
    axs[1].legend()

    axs[2].plot(times_obs, normalize(flux_obs) - normalize(lc))
    axs[2].tick_params(labeltop=False)
    
    axs[3].plot(times_obs, normalize(sin), label="simulated lightcurve", color = "xkcd:magenta")
    axs[3].legend()

    axs[4].plot(times_obs, normalize(flux_obs) - normalize(sin))
    axs[4].tick_params(labeltop=False)

    st.pyplot(fig)

st.title('Photo-clouds')

tab1, tab2 = st.tabs(["Light Curve", "Animation"])

with tab1:
    st.header("Light Curve")
    
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    
    if uploaded_file is not None:
        try:
            times, flux = load_lines(uploaded_file)
            plot(times, flux)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
    
    # Move user inputs to the sidebar
    st.sidebar.title("Controls")
    
    # Sidebar user inputs for the parameters
    length = st.sidebar.slider('Length of Observation [days]', min_value=1, max_value=365*2, value=12, step=1)
    num_observations = st.sidebar.slider('Number of Observations', min_value=10, max_value=10000, value=5000, step=10)
    period = st.sidebar.slider('Period [hrs]', min_value=0.1, max_value=100.0, value=9.8, step=0.1)
    phase = st.sidebar.slider('Phase [radians]', min_value=0.0, max_value=2 * np.pi, value=0.0, step=0.1)
    y_noise_std = st.sidebar.slider('y noise std', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    t_noise_std = st.sidebar.slider('t noise std', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    day_fraction = st.sidebar.slider('Day/Night Duty Cycle', min_value=0.05, max_value=1.0, value=0.5, step=0.05)
    irregular = st.sidebar.checkbox('Irregular Spacing', value=False)
    logy = st.sidebar.checkbox('log power', value=False)

with tab2:
    st.header("Animation")
    
    longitudes = np.random.uniform(-np.pi, np.pi, size=5)
    latitudes = np.random.uniform(-np.pi/2, np.pi/2, size=5)
    t = np.linspace(0, 10, 50)
    
    # Run animation
    ani = run_animation(longitudes.copy(), latitudes.copy(), t)
    
    # Render as HTML
    html_anim = ani.to_jshtml()
    st.components.v1.html(html_anim, height=400)