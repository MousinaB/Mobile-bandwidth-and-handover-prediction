#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import pandas as pd


# In[2]:


# List of dictionaries defining transition probabilities between states ('Good', 'Medium', 'Bad') 
# based on value change ranges. Each entry specifies:
# - a 'range' of change,
# - the 'previous' state,
# - the 'current' (next) state,
# - and the likelihood ('probability') of that transition occurring.

transition_probabilities_large = [
    {"range": (0, 5), "previous": "Good", "current": "Good", "probability": 65},
    {"range": (-15, -5), "previous": "Good", "current": "Medium", "probability": 25},
    {"range": (-25, -10), "previous": "Good", "current": "Bad", "probability": 10},
    {"range": (5, 10), "previous": "Medium", "current": "Good", "probability": 20},
    {"range": (0, 2), "previous": "Medium", "current": "Medium", "probability": 65},
    {"range": (-15, -5), "previous": "Medium", "current": "Bad", "probability": 15},
    {"range": (10, 25), "previous": "Bad", "current": "Good", "probability": 15},
    {"range": (5, 10), "previous": "Bad", "current": "Medium", "probability": 25},
    {"range": (-2, 0), "previous": "Bad", "current": "Bad", "probability": 60},
]


# In[3]:


# Transition probabilities for the "small" scenario.
# Each dictionary represents a possible state transition (Good, Medium, Bad),
# determined by a range of value changes, and the likelihood of that transition.
# These are likely used in a state machine or time series model to simulate or predict transitions.


transition_probabilities_small = [
    {"range": (-5, 0), "previous": "Good", "current": "Good", "probability": 65},
    {"range": (5, 15), "previous": "Good", "current": "Medium", "probability": 25},
    {"range": (10,25), "previous": "Good", "current": "Bad", "probability": 10},
    {"range": (-10, -5), "previous": "Medium", "current": "Good", "probability": 20},
    {"range": (-2, 0), "previous": "Medium", "current": "Medium", "probability": 65},
    {"range": (5, 15), "previous": "Medium", "current": "Bad", "probability": 15},
    {"range": (-25, -10), "previous": "Bad", "current": "Good", "probability": 15},
    {"range": (-10, -5), "previous": "Bad", "current": "Medium", "probability": 25},
    {"range": (0, 2), "previous": "Bad", "current": "Bad", "probability": 60},]



# In[4]:


# Functions to classify RSSI (Received Signal Strength Indicator) values into quality categories:
# - 'Good' for strong signals between -55 and -40 dBm,
# - 'Medium' for moderate signals between -80 and -55 dBm,
# - 'Bad' for weak signals between -100 and -80 dBm.
# The `categories_rssi` dictionary maps each category to its corresponding condition function.


def in_good_range(rssi):
    return -55 <= rssi <= -40

def in_medium_range(rssi):
    return -80 <= rssi < -55

def in_bad_range(rssi):
    return -100 <= rssi < -80

categories_rssi = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}


# In[5]:


# Functions to classify RSRQ (Reference Signal Received Quality) values into quality categories:
# - 'Good' for strong signal quality between -6 and -3 dB,
# - 'Medium' for moderate quality between -15 and -6 dB,
# - 'Bad' for poor quality between -20 and -15 dB.
# The `categories_rsrq` dictionary maps each category to its corresponding condition function.


def in_good_range(rsrq):
    return -6 <= rsrq <= -3

def in_medium_range(rsrq):
    return -15 <= rsrq < -6

def in_bad_range(rsrq):
    return -20 <= rsrq < -15

categories_rsrq = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}


# In[6]:


# Functions to classify speed values (possibly in km/h or m/s) into quality categories:
# - 'Good' for low speeds between 5 and 15 units,
# - 'Medium' for moderate speeds between 15 and 40 units,
# - 'Bad' for high speeds between 40 and 50 units.
# The `categories_speed` dictionary maps each category to its corresponding condition function.

def in_good_range(speed):
    return 5 <= speed <= 15

def in_medium_range(speed):
    return 15 < speed <= 40

def in_bad_range(speed):
    return 40 < speed <=50

categories_speed = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}


# In[7]:


# Functions to classify TA (Timing Advance) values into quality categories:
# - 'Good' for TA values between 10 and 50 units (closer proximity to base station),
# - 'Medium' for TA values between 50 and 140 units,
# - 'Bad' for TA values between 140 and 200 units (farther from the base station).
# The `categories_ta` dictionary maps each category to its corresponding condition function.


def in_good_range(ta):
    return 10 <= ta <= 50

def in_medium_range(ta):
    return 50 < ta <= 140

def in_bad_range(ta):
    return 140 < ta <=200

categories_ta = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}


# In[8]:


# Functions to classify the number of LTE Neighbouring cells into quality categories:
# - 'Good' when the number of neighbours is between 1 and 2 (suggesting optimal handover conditions),
# - 'Medium' when the number of neighbours is between 2 and less than 4,
# - 'Bad' when the number of neighbours is between 4 and 6 (may indicate interference or dense environment).
# The `categories_LTE_Neighbour` dictionary maps each category to its corresponding condition function.

def in_good_range(LTE_Neighbour):
    return 1 <= LTE_Neighbour <= 2

def in_medium_range(LTE_Neighbour):
    return 2 <= LTE_Neighbour < 4

def in_bad_range(LTE_Neighbour):
    return 4 <= LTE_Neighbour < 6

categories_LTE_Neighbour = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}


# In[9]:


# Function to classify an RSSI value into 'Good', 'Medium', or 'Bad' using the predefined `categories_rssi` mapping.
# Iterates over each category and applies the corresponding range check.
# Returns the matching category or None if no category matches.


def classify_rssi(rssi):
    for category, check_function in categories_rssi.items():
        if check_function(rssi):
            return category
    return None

# Function to calculate a parameter 'a' based on the RSSI value.
# It linearly maps RSSI from the range [-100, -40] to a value between 5 and 50.
# This is likely used as a scaled weight or importance metric for signal strength.

def calculate_a(rssi):
    return 5 + (rssi + 100) * (50 - 5) / (100 - 40)


# In[10]:


# Function to classify an RSRQ (Reference Signal Received Quality) value into 'Good', 'Medium', or 'Bad'
# based on predefined category ranges in `categories_rsrq`.


def classify_rsrq(rsrq):
    for category, check_function in categories_rsrq.items():
        if check_function(rsrq):
            return category
    return None

# Function to calculate parameter 'b' based on the RSRQ value.
# This linearly maps RSRQ from the range [-20, -3] to a value between 5 and 50.
# The transformation is useful for scaling RSRQ values into a normalized score or weight.

def calculate_b(rsrq):
    return 5 + (rsrq + 20) * (50 - 5) / (20 - 3)


# In[11]:


#Function to classify a speed value into 'High', 'Medium', or 'Low'
# based on predefined category ranges in `categories_speed`.


def classify_speed(speed):
    for category, check_function in categories_speed.items():
        if check_function(speed):
            return category
    return None

# Function to calculate parameter 'c' based on the speed value.
# This linearly maps speed from the range [0, 50] to a value between 50 and 5 (inversely proportional).
# The transformation is useful when higher speeds are given lower scores, e.g., for mobility cost metrics.

def calculate_c(speed):
    return 5 + (50-speed) * (50 - 5) / (50 - 5)


# In[12]:


# Function to classify a TA (Timing Advance) value into 'Near', 'Medium', or 'Far'
# based on predefined category ranges in `categories_ta`.

def classify_ta(ta):
    for category, check_function in categories_ta.items():
        if check_function(ta):
            return category
    return None

# Function to calculate parameter 'd' based on the TA value.
# This linearly maps TA from the range [10, 200] to a value between 50 and 5 (inversely proportional).
# The transformation is useful when larger TA values (indicating greater distance from the base station)
# are assigned lower scores or weights.

def calculate_d(ta):
    return 5 + (200-ta) * (50 - 5) / (200 - 10)


# In[13]:


# Function to classify the number of LTE Neighbours into categories like 'Dense', 'Moderate', or 'Sparse'
# based on predefined category ranges in `categories_LTE_Neighbour`.


def classify_LTE_Neighbour(LTE_Neighbour):
    for category, check_function in categories_LTE_Neighbour.items():
        if check_function(LTE_Neighbour):
            return category
    return None


#Function to calculate parameter 'e' based on the number of LTE Neighbours.
# This linearly maps the LTE Neighbour count from the range [1, 6] to a value between 5 and 50.
# A small random noise in the range [-0.1, 0.1] is added to introduce slight variation.
# This is useful in scenarios where deterministic mapping could lead to uniform results or artifacts.

def calculate_e(LTE_Neighbour):
    return 5 + (LTE_Neighbour -6) * (50 - 5) / (6 - 1) + random.uniform(-0.1, 0.1)


# In[14]:


# Function to calculate parameter 'f' based on the LTE band value.
# Returns a random integer within a specific range depending on the band:
# - Band 0: returns a value between 5 and 15
# - Band 1: returns a value between 15 and 30
# - Any other band: returns a value between 30 and 50
# This is useful when different frequency bands are associated with varying quality, cost, or performance.


def calculate_f(band):
    if band == 0:
        return random.randint(5, 15)
    elif band == 1:
        return random.randint(15, 30)
    else:
        return random.randint(30, 50)


# In[15]:


# Function to calculate parameter 'g' based on the ECHNG (eNodeB change count or handover attempts).
# - If ECHNG is less than or equal to 5, it returns a random value between -20 and -5,
#   indicating a higher penalty due to frequent changes.
# - Otherwise, it returns a small positive value between 0 and 3,
#   representing relatively stable connectivity.
# This transformation can be useful for penalizing unstable network conditions.

def calculate_g(echng):
    if echng <= 5:
        return random.randint(-20, -5) 
    else:
        return random.randint(0, 3) 


# In[16]:


# Function to fetch the transition range for a given state transition in a large transition matrix.
# - It searches for a transition entry in `transition_probabilities_large` where both the
#   'previous' and 'current' states match the input.
# - If found, it returns the associated 'range' value, which might represent a probability range or score.
# - If no matching transition is found, it raises a ValueError with the corresponding state information.

def pick_transition_large(previous, current):
    for transition in transition_probabilities_large:
        if transition["previous"] == previous and transition["current"] == current:
            return transition["range"]
    raise ValueError(f"No transition found for ({previous}, {current})")


# In[17]:


# Function to fetch the transition range for a given state transition in a small transition matrix.
# - It looks for a transition in `transition_probabilities_small` where both 'previous' and 'current' states match the input.
# - If a match is found, it returns the associated 'range' value (e.g., a probability range or score).
# - If no matching transition is found, it raises a ValueError with details of the missing transition.

def pick_transition_small(previous, current):
    for transition in transition_probabilities_small:
        if transition["previous"] == previous and transition["current"] == current:
            return transition["range"]
    raise ValueError(f"No transition found for ({previous}, {current})")


# In[18]:


# Function to calculate bandwidth (bt) values over a given time period.
# The calculation is influenced by multiple weighted parameters like RSSI, RSRQ, Speed, TA, etc.
# Inputs:
# - total_seconds: Total number of time steps to compute.
# - w_*: Weights for each parameter (RSSI, RSRQ, Speed, TA, LTE Neighbour, Band, ECHNG, previous Bandwidth).
# - *_values: Lists or arrays containing precomputed values for each parameter at each time step.
#
# At each time step:
# - For the first second (i = 0), bandwidth is computed purely from the weighted sum of all input parameters.
# - For subsequent seconds, it uses a combination of the previous bandwidth (weighted by w_bandwidth)
#   and the current parameter values (weighted sum).
# - The resulting bandwidth value is clipped between 1 and 49 to keep it within valid bounds.


def calculate_bandwidth(total_seconds, w_rssi,w_rsrq,w_speed,w_ta,w_LTE_Neighbour,w_band,w_echng, w_bandwidth, a_values,b_values,c_values,d_values,e_values,f_values,g_values):
    bt = np.zeros(total_seconds)
    for i in range(total_seconds):
        if i == 0:
            bt[i] = w_rssi * a_values[i]+w_rsrq * b_values[i]+w_speed*c_values[i]+w_ta*d_values[i]+w_LTE_Neighbour*e_values[i]+w_band*f_values[i]+w_echng*g_values[i]
        else:
            bt[i] = w_bandwidth * bt[i - 1] + (w_rssi * a_values[i]+w_rsrq * b_values[i]+w_speed*c_values[i]+w_ta*d_values[i]+w_LTE_Neighbour*e_values[i]+w_band*f_values[i]+w_echng*g_values[i])
        bt[i] = np.clip(bt[i], 1, 49)
    return bt


# In[19]:


# Function to generate simulated time-series data for network parameters
# It models dynamic transitions using transition probabilities and computes final bandwidth
# Inputs:
# - total_seconds: duration for which to simulate the data
# - weights: coefficients for each feature's contribution to bandwidth (with sensible defaults)

def generate_data(total_seconds, w_rssi=0.11,w_rsrq=0.04,w_speed=0.02,w_ta=0.05,w_LTE_Neighbour=0.02,w_band=0.17,w_echng=0.01, w_bandwidth=0.58):
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize parameters with random values within expected real-world ranges
    rssi = random.uniform(-100, -40) 
    rsrq = random.uniform(-20, -3) 
    speed = random.uniform(5,50)
    ta = random.uniform(10,200)
    LTE_Neighbour = random.uniform(1,6)
    
    
    # Lists to store raw feature values
    rssi_values = []
    rsrq_values = []
    speed_values=[]
    ta_values = []
    LTE_Neighbour_values = []
    band_values = []
    echng_values = []
    

    
    # Lists to store transformed values A-G (for bandwidth calculation)
    a_values = []
    b_values = []
    c_values = []
    d_values = []
    e_values = []
    f_values = []
    g_values = []

    
    
    # Generate values for each second
    for i in range(total_seconds):
        
        # Calculate transformed metrics based on current values
        a = round(calculate_a(rssi), 2)  
        b = round(calculate_b(rsrq), 2)
        c = round(calculate_c(speed), 2)
        d = round(calculate_d(ta), 2)
        e = round(calculate_e(LTE_Neighbour), 2)
        band = random.choices([0, 1, 2], weights=[1, 1, 1])[0]  
        f = round(calculate_f(band), 2)
       
        
        
        rssi = round(rssi, 2)  
        rsrq = round(rsrq, 2) 
        speed = round(speed, 2) 
        ta = round(ta, 2)
        LTE_Neighbour = round(LTE_Neighbour, 2)
        echng = random.randint(0, 99)
        
        
        
        a_values.append(a)
        b_values.append(b)
        c_values.append(c)
        d_values.append(d)
        e_values.append(e)
        f_values.append(f)
        
        
        # Store raw and transformed values
        rssi_values.append(rssi)
        rsrq_values.append(rsrq)
        speed_values.append(speed)
        ta_values.append(ta)
        LTE_Neighbour_values.append(LTE_Neighbour)
        band_values.append(band)
        echng_values.append(echng)
        g = calculate_g(echng)
        g_values.append(g)
        
        
        # Simulate transitions for each feature using classification and transition ranges
        previous_category_rssi = classify_rssi(rssi)
        previous_category_rsrq = classify_rsrq(rsrq)
        previous_category_speed = classify_speed(speed)
        previous_category_ta = classify_ta(ta)
        previous_category_LTE_Neighbour = classify_LTE_Neighbour(LTE_Neighbour)
        
         # RSSI transition
        if previous_category_rssi:
            next_category_rssi = random.choices(
                ["Good", "Medium", "Bad"],
                weights=[65 if previous_category_rssi == "Good" else 20,
                         25 if previous_category_rssi == "Good" else 65,
                         10 if previous_category_rssi == "Good" else 15]
            )[0]

            try:
                rssi_range = pick_transition_large(previous_category_rssi, next_category_rssi)
                value = round(random.uniform(*rssi_range), 2)  
                rssi += value
                rssi = round(np.clip(rssi, -100, -40), 2)  
            except ValueError:
                pass  

            
            
        # RSRQ transition    
        if previous_category_rsrq:
            next_category_rsrq = random.choices(
                ["Good", "Medium", "Bad"],
                weights=[65 if previous_category_rsrq == "Good" else 20,
                         25 if previous_category_rsrq == "Good" else 65,
                         10 if previous_category_rsrq == "Good" else 15]
            )[0]

            try:
                rsrq_range = pick_transition_large(previous_category_rsrq, next_category_rsrq)
                value = round(random.uniform(*rsrq_range), 2)  
                rsrq += value
                rsrq = round(np.clip(rsrq, -20, -3), 2)  
            except ValueError:
                pass  
            
        # Speed transition   
        if previous_category_speed:
            next_category_speed = random.choices(
                ["Good", "Medium", "Bad"],
                weights=[65 if previous_category_speed == "Good" else 20,
                         25 if previous_category_speed == "Good" else 65,
                         10 if previous_category_speed == "Good" else 15]
            )[0]

            try:
                speed_range = pick_transition_large(previous_category_speed, next_category_speed)
                value = round(random.uniform(*speed_range), 2)  
                speed += value
                speed = round(np.clip(speed, 5, 50), 2)  
            except ValueError:
                pass 
         
        # TA transition   
        if previous_category_ta:
            next_category_ta = random.choices(
                ["Good", "Medium", "Bad"],
                weights=[65 if previous_category_ta == "Good" else 20,
                         25 if previous_category_ta == "Good" else 65,
                         10 if previous_category_ta == "Good" else 15]
            )[0]

            try:
                ta_range = pick_transition_large(previous_category_ta, next_category_ta)
                value = round(random.uniform(*ta_range), 2)  
                ta += value
                ta = round(np.clip(ta, 10, 200), 2)  
            except ValueError:
                pass  

        # LTE Neighbour transition   
        if previous_category_LTE_Neighbour:
            next_category_LTE_Neighbour = random.choices(["Good", "Medium", "Bad"],
            weights=[65 if previous_category_LTE_Neighbour == "Good" else 20,
                     25 if previous_category_LTE_Neighbour == "Good" else 65,
                     10 if previous_category_LTE_Neighbour == "Good" else 15])[0]

            try:
                LTE_Neighbour_range = pick_transition_large(previous_category_LTE_Neighbour, next_category_LTE_Neighbour)
                value = round(random.uniform(*LTE_Neighbour_range) * random.choice([-1, 1]), 2)
                LTE_Neighbour += value + random.uniform(-0.2, 0.2)  
                LTE_Neighbour = max(1, min(LTE_Neighbour, 6))
            except ValueError:
                pass 
            
    
    # Compute bandwidth for all time steps based on weighted sum of A-G values
    bandwidth_values = calculate_bandwidth(total_seconds, w_rssi,w_rsrq,w_speed,w_ta,w_LTE_Neighbour,w_band,w_echng, w_bandwidth, a_values,b_values,c_values,d_values,e_values,f_values,g_values)
    
    # Compile all values into a Pandas DataFrame
    data = {"RSSI": rssi_values,"RSRQ": rsrq_values,"Speed":speed_values,"TA":ta_values,"LTE_Neighbour":LTE_Neighbour_values,"Band":band_values,"Echng": echng_values, "A": a_values,"B": b_values,"C":c_values,"D":d_values,"E":e_values,"F":f_values,"G": g_values, "Bandwidth": np.round(bandwidth_values, 2)}
    return pd.DataFrame(data)


# In[29]:


df = generate_data(total_seconds=300*60*60)
df.to_csv("Bandwidth_synthetic_data_1.csv", index=False)
print("Data generated and saved to 'Bandwidth_synthetic_data_1.csv'.")


# In[30]:


get_ipython().system('pip install scikit-learn')


# In[31]:


df = pd.read_csv("Bandwidth_synthetic_data_1.csv")  
print(df.columns)


# In[32]:


df.head()


# In[33]:


# Calculate statistics for the "Bandwidth" column
bandwidth_avg = df["Bandwidth"].mean()
bandwidth_median = df["Bandwidth"].median()
bandwidth_max = df["Bandwidth"].max()
bandwidth_min = df["Bandwidth"].min()
bandwidth_std = df["Bandwidth"].std()
bandwidth_length = len(df["Bandwidth"])

# Print the results
print(f"Average Bandwidth: {bandwidth_avg:.4f}")
print(f"Median Bandwidth: {bandwidth_median:.4f}")
print(f"Maximum Bandwidth: {bandwidth_max:.4f}")
print(f"Minimum Bandwidth: {bandwidth_min:.4f}")
print(f"Standard Deviation of Bandwidth: {bandwidth_std:.4f}")
print(f"Length of Bandwidth column: {bandwidth_length}")


# In[34]:


df.isnull()


# In[35]:


df.dropna()


# In[36]:


print(df.isna().sum())  # Check for NaN values in the entire dataset


# In[37]:


df.mean()


# # Seed Removed:
#    The lines np.random.seed(42) and random.seed(42) have been removed and replaced with np.random.seed(None) and random.seed(None). This will remove any fixed seeding, leading to fully random results on each run.
# 
# # Modified Weights: 
#    Again   I changed the weight values for parameters like w_rssi, w_rsrq, etc., in the generate_new_data() function to get new data.

# In[41]:


# Function to generate simulated time-series data for network parameters
# It models dynamic transitions using transition probabilities and computes final bandwidth
# Inputs:
# - total_seconds: duration for which to simulate the data
# - weights: coefficients for each feature's contribution to bandwidth (with sensible defaults)

def generate_data(total_seconds, w_rssi=0.11,w_rsrq=0.04,w_speed=0.02,w_ta=0.05,w_LTE_Neighbour=0.02,w_band=0.17,w_echng=0.01, w_bandwidth=0.58):
    # Set seeds for reproducibility
    np.random.seed(None)
    random.seed(None)
    
    # Initialize parameters with random values within expected real-world ranges
    rssi = random.uniform(-100, -40) 
    rsrq = random.uniform(-20, -3) 
    speed = random.uniform(5,50)
    ta = random.uniform(10,200)
    LTE_Neighbour = random.uniform(1,6)
    
    
    # Lists to store raw feature values
    rssi_values = []
    rsrq_values = []
    speed_values=[]
    ta_values = []
    LTE_Neighbour_values = []
    band_values = []
    echng_values = []
    

    
    # Lists to store transformed values A-G (for bandwidth calculation)
    a_values = []
    b_values = []
    c_values = []
    d_values = []
    e_values = []
    f_values = []
    g_values = []

    
    
    # Generate values for each second
    for i in range(total_seconds):
        
        # Calculate transformed metrics based on current values
        a = round(calculate_a(rssi), 2)  
        b = round(calculate_b(rsrq), 2)
        c = round(calculate_c(speed), 2)
        d = round(calculate_d(ta), 2)
        e = round(calculate_e(LTE_Neighbour), 2)
        band = random.choices([0, 1, 2], weights=[1, 1, 1])[0]  
        f = round(calculate_f(band), 2)
       
        
        
        rssi = round(rssi, 2)  
        rsrq = round(rsrq, 2) 
        speed = round(speed, 2) 
        ta = round(ta, 2)
        LTE_Neighbour = round(LTE_Neighbour, 2)
        echng = random.randint(0, 99)
        
        
        
        a_values.append(a)
        b_values.append(b)
        c_values.append(c)
        d_values.append(d)
        e_values.append(e)
        f_values.append(f)
        
        
        # Store raw and transformed values
        rssi_values.append(rssi)
        rsrq_values.append(rsrq)
        speed_values.append(speed)
        ta_values.append(ta)
        LTE_Neighbour_values.append(LTE_Neighbour)
        band_values.append(band)
        echng_values.append(echng)
        g = calculate_g(echng)
        g_values.append(g)
        
        
        # Simulate transitions for each feature using classification and transition ranges
        previous_category_rssi = classify_rssi(rssi)
        previous_category_rsrq = classify_rsrq(rsrq)
        previous_category_speed = classify_speed(speed)
        previous_category_ta = classify_ta(ta)
        previous_category_LTE_Neighbour = classify_LTE_Neighbour(LTE_Neighbour)
        
         # RSSI transition
        if previous_category_rssi:
            next_category_rssi = random.choices(
                ["Good", "Medium", "Bad"],
                weights=[65 if previous_category_rssi == "Good" else 20,
                         25 if previous_category_rssi == "Good" else 65,
                         10 if previous_category_rssi == "Good" else 15]
            )[0]

            try:
                rssi_range = pick_transition_large(previous_category_rssi, next_category_rssi)
                value = round(random.uniform(*rssi_range), 2)  
                rssi += value
                rssi = round(np.clip(rssi, -100, -40), 2)  
            except ValueError:
                pass  

            
            
        # RSRQ transition    
        if previous_category_rsrq:
            next_category_rsrq = random.choices(
                ["Good", "Medium", "Bad"],
                weights=[65 if previous_category_rsrq == "Good" else 20,
                         25 if previous_category_rsrq == "Good" else 65,
                         10 if previous_category_rsrq == "Good" else 15]
            )[0]

            try:
                rsrq_range = pick_transition_large(previous_category_rsrq, next_category_rsrq)
                value = round(random.uniform(*rsrq_range), 2)  
                rsrq += value
                rsrq = round(np.clip(rsrq, -20, -3), 2)  
            except ValueError:
                pass  
            
        # Speed transition   
        if previous_category_speed:
            next_category_speed = random.choices(
                ["Good", "Medium", "Bad"],
                weights=[65 if previous_category_speed == "Good" else 20,
                         25 if previous_category_speed == "Good" else 65,
                         10 if previous_category_speed == "Good" else 15]
            )[0]

            try:
                speed_range = pick_transition_large(previous_category_speed, next_category_speed)
                value = round(random.uniform(*speed_range), 2)  
                speed += value
                speed = round(np.clip(speed, 5, 50), 2)  
            except ValueError:
                pass 
         
        # TA transition   
        if previous_category_ta:
            next_category_ta = random.choices(
                ["Good", "Medium", "Bad"],
                weights=[65 if previous_category_ta == "Good" else 20,
                         25 if previous_category_ta == "Good" else 65,
                         10 if previous_category_ta == "Good" else 15]
            )[0]

            try:
                ta_range = pick_transition_large(previous_category_ta, next_category_ta)
                value = round(random.uniform(*ta_range), 2)  
                ta += value
                ta = round(np.clip(ta, 10, 200), 2)  
            except ValueError:
                pass  

        # LTE Neighbour transition   
        if previous_category_LTE_Neighbour:
            next_category_LTE_Neighbour = random.choices(["Good", "Medium", "Bad"],
            weights=[65 if previous_category_LTE_Neighbour == "Good" else 20,
                     25 if previous_category_LTE_Neighbour == "Good" else 65,
                     10 if previous_category_LTE_Neighbour == "Good" else 15])[0]

            try:
                LTE_Neighbour_range = pick_transition_large(previous_category_LTE_Neighbour, next_category_LTE_Neighbour)
                value = round(random.uniform(*LTE_Neighbour_range) * random.choice([-1, 1]), 2)
                LTE_Neighbour += value + random.uniform(-0.2, 0.2)  
                LTE_Neighbour = max(1, min(LTE_Neighbour, 6))
            except ValueError:
                pass 
            
    
    # Compute bandwidth for all time steps based on weighted sum of A-G values
    bandwidth_values = calculate_bandwidth(total_seconds, w_rssi,w_rsrq,w_speed,w_ta,w_LTE_Neighbour,w_band,w_echng, w_bandwidth, a_values,b_values,c_values,d_values,e_values,f_values,g_values)
    
    # Compile all values into a Pandas DataFrame
    data = {"RSSI": rssi_values,"RSRQ": rsrq_values,"Speed":speed_values,"TA":ta_values,"LTE_Neighbour":LTE_Neighbour_values,"Band":band_values,"Echng": echng_values, "A": a_values,"B": b_values,"C":c_values,"D":d_values,"E":e_values,"F":f_values,"G": g_values, "Bandwidth": np.round(bandwidth_values, 2)}
    return pd.DataFrame(data)


# In[42]:


df = generate_data(total_seconds=300*60*60)
df.to_csv("Bandwidth_synthetic_data_2.csv", index=False)
print("Data generated and saved to 'Bandwidth_synthetic_data_2.csv'.")


# In[43]:


dff = pd.read_csv("Bandwidth_synthetic_data_2.csv")  

# Calculate statistics for the "Bandwidth" column
bandwidth_avg = dff["Bandwidth"].mean()
bandwidth_median = dff["Bandwidth"].median()
bandwidth_max = dff["Bandwidth"].max()
bandwidth_min = dff["Bandwidth"].min()
bandwidth_std = dff["Bandwidth"].std()
bandwidth_length = len(dff["Bandwidth"])

# Print the results
print(f"Average Bandwidth: {bandwidth_avg:.4f}")
print(f"Median Bandwidth: {bandwidth_median:.4f}")
print(f"Maximum Bandwidth: {bandwidth_max:.4f}")
print(f"Minimum Bandwidth: {bandwidth_min:.4f}")
print(f"Standard Deviation of Bandwidth: {bandwidth_std:.4f}")
print(f"Length of Bandwidth column: {bandwidth_length}")


# In[44]:


dff.mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




