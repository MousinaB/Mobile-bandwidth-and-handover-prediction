#!/usr/bin/env python
# coding: utf-8

# In[762]:


import numpy as np
import random
import pandas as pd


# In[763]:


# Transition probabilities and ranges for DL
transition_probabilities_large = [
    {"range": (0, 5), "previous": "Good", "current": "Good", "probability": 50},
    {"range": (-15, -5), "previous": "Good", "current": "Medium", "probability": 25},
    {"range": (-30, -15), "previous": "Good", "current": "Bad", "probability": 25},

    {"range": (10, 20), "previous": "Medium", "current": "Good", "probability": 25},
    {"range": (0, 5), "previous": "Medium", "current": "Medium", "probability": 50},
    {"range": (-20, -10), "previous": "Medium", "current": "Bad", "probability": 25},

    {"range": (30, 50), "previous": "Bad", "current": "Good", "probability": 50},
    {"range": (10, 20), "previous": "Bad", "current": "Medium", "probability": 25},
    {"range": (-5, 0), "previous": "Bad", "current": "Bad", "probability": 25},
]


# In[764]:


# Transition probabilities and ranges for DL
transition_probabilities_small = [
    {"range": (-5,0), "previous": "Good", "current": "Good", "probability": 50},
    {"range": (5,15), "previous": "Good", "current": "Medium", "probability": 25},
    {"range": (15,30), "previous": "Good", "current": "Bad", "probability": 25},

    {"range": (-20,-10), "previous": "Medium", "current": "Good", "probability": 25},
    {"range": (-5,0), "previous": "Medium", "current": "Medium", "probability": 50},
    {"range": (10,20), "previous": "Medium", "current": "Bad", "probability": 25},

    {"range": (-50,-30), "previous": "Bad", "current": "Good", "probability": 50},
    {"range": (-20,-10), "previous": "Bad", "current": "Medium", "probability": 25},
    {"range": (0,5), "previous": "Bad", "current": "Bad", "probability": 25},
]


# In[765]:


# DL ranges for classification
def in_good_range(DL):
    return -70 <= DL < -40  # Corrected Good range

def in_medium_range(DL):
    return -90 <= DL < -70  # Medium range

def in_bad_range(DL):
    return -100 <= DL < -90  # Bad range

# Mapping categories to classification functions
categories_DL = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}

# Classify DL into Good, Medium, or Bad
def classify_DL(DL):
    for category, check_function in categories_DL.items():
        if check_function(DL):
            return category
    return None  # Return None if out of range



# In[766]:


def in_good_range(UL):
    return 1 <= UL < 5

def in_medium_range(UL):
    return 4 <= UL < 11

def in_bad_range(UL):
    return 11 <= UL <= 15

categories_UL = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}

def classify_UL(UL):
    for category, check_function in categories_UL.items():
        if check_function(UL):
            return category
    return None


# In[767]:


def in_good_range(rssi):
    return -55 <= rssi <= -40

def in_medium_range(rssi):
    return -80 <= rssi < -55

def in_bad_range(rssi):
    return -100 <= rssi < -80

categories_rssi = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}

def classify_rssi(rssi):
    for category, check_function in categories_rssi.items():
        if check_function(rssi):
            return category
    return None


# In[768]:


def in_good_range(rsrq):
    return -6 <= rsrq <= -3

def in_medium_range(rsrq):
    return -15 <= rsrq < -6

def in_bad_range(rsrq):
    return -20 <= rsrq < -15

categories_rsrq = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}

def classify_rsrq(rsrq):
    for category, check_function in categories_rsrq.items():
        if check_function(rsrq):
            return category
    return None


# In[769]:


def in_good_range(rsrp):
    return -120 <= rsrp < -100

def in_medium_range(rsrp):
    return -100 <= rsrp < -50

def in_bad_range(rsrp):
    return -50 <= rsrp < -30

categories_rsrp = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}

def classify_rsrp(rsrp):
    for category, check_function in categories_rsrp.items():
        if check_function(rsrp):
            return category
    return None


# In[770]:


def in_good_range(nrxsrp):
    return -120 <= nrxsrp < -100

def in_medium_range(nrxsrp):
    return -100 <= nrxsrp < -50

def in_bad_range(nrxsrp):
    return -50 <= nrxsrp < -30

categories_nrxsrp = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}

def classify_nrxsrp(nrxsrp):
    for category, check_function in categories_nrxsrp.items():
        if check_function(nrxsrp):
            return category
    return None


# In[771]:


def in_good_range(nrxsrq):
    return 5 <= nrxsrq  <= 15

def in_medium_range(nrxsrq):
    return 15 < nrxsrq < 40

def in_bad_range(nrxsrq):
    return 40 <= nrxsrq <=50

categories_nrxsrq = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}

def classify_nrxsrq(nrxsrq):
    for category, check_function in categories_nrxsrq.items():
        if check_function(nrxsrq):
            return category
    return None


# In[772]:


def in_good_range(snr):
    return 1 <= snr <= 6

def in_medium_range(snr):
    return 6 < snr <14

def in_bad_range(snr):
    return 14<= snr<=20

categories_snr = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}

def classify_snr(snr):
    for category, check_function in categories_snr.items():
        if check_function(snr):
            return category
    return None


# In[773]:


def in_good_range(CQI):
    return 10 <= CQI < 15

def in_medium_range(CQI):
    return 5 <= CQI < 10

def in_bad_range(CQI):
    return 0 <= CQI < 5



categories_CQI = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}

def classify_CQI(CQI):
    if not isinstance(CQI, (int, float)) or CQI < 0 or CQI > 15:
        return "Invalid CQI"
    for category, check_function in categories_CQI.items():
        if check_function(CQI):
            return category
    return "Unknown"



# In[774]:


def in_good_range(speed):
    return 5 <= speed <= 15

def in_medium_range(speed):
    return 15 < speed < 40

def in_bad_range(speed):
    return 40 <= speed <=50

categories_speed = {"Good": in_good_range, "Medium": in_medium_range, "Bad": in_bad_range}

def classify_speed(speed):
    for category, check_function in categories_speed.items():
        if check_function(speed):
            return category
    return None


# In[775]:


# Pick transition range based on previous and current state
def pick_transition_large(previous, current):
    for transition in transition_probabilities_large:
        if transition["previous"] == previous and transition["current"] == current:
            return transition["range"]
    return (-5, 5)  # Default small fluctuation if no transition found


# In[776]:


# Pick transition range based on previous and current state
def pick_transition_small(previous, current):
    for transition in transition_probabilities_small:
        if transition["previous"] == previous and transition["current"] == current:
            return transition["range"]
    return (-5, 5)  # Default small fluctuation if no transition found


# In[777]:


# Calculate handover based on transition flags
def calculate_handover(total_seconds,w_DL,w_UL, w_rssi,w_rsrq,w_rsrp,w_nrxsrp,w_nrxsrq,w_snr,w_CQI,w_speed,transition_flag_DL,transition_flag_UL,transition_flag_rssi,transition_flag_rsrq,transition_flag_rsrp,transition_flag_nrxsrp,transition_flag_nrxsrq,transition_flag_snr,transition_flag_CQI,transition_flag_speed):
    ho = np.zeros(total_seconds)
    for i in range(total_seconds):
        ho[i] =w_DL*transition_flag_DL[i]+w_UL*transition_flag_UL[i]+w_rssi*transition_flag_rssi[i]+w_rsrq*transition_flag_rsrq[i]+w_rsrp*transition_flag_rsrp[i]+w_nrxsrp * transition_flag_nrxsrp[i]+w_nrxsrq * transition_flag_nrxsrq[i]+w_snr * transition_flag_snr[i]+w_CQI*transition_flag_CQI[i]+w_speed*transition_flag_speed[i]
    return [1 if h > 0.7 else 0 for h in ho]


# In[758]:


# Generate synthetic DL data and handover status
def generate_data(total_seconds, w_DL=0.12,w_UL=0.12, w_rssi=0.12,w_rsrq=0.12,w_rsrp=0.12,w_nrxsrp=0.12,w_nrxsrq=0.12,w_snr=0.12,w_CQI=0.47,w_speed=0.12):
    np.random.seed(42)
    random.seed(42)

    
    DL = random.uniform(5,50)
    UL = random.uniform(1,15)
    rssi = random.uniform(-100, -40)
    rsrq = random.uniform(-20, -3)
    rsrp = random.uniform(-120, -30)
    nrxsrp=random.uniform(-120, -30)
    nrxsrq=random.uniform(-20, -3)
    snr=random.uniform(1,20)
    CQI=random.uniform(0,15)
    speed=random.uniform(5,50)
    
    
    
    
    DL_values = []
    UL_values = []
    rssi_values = []
    rsrq_values = []
    rsrp_values = []
    nrxsrp_values = []
    nrxsrq_values = []
    snr_values = []
    CQI_values = []
    speed_values = []
    
    
    
    
    transition_flag_DL=[]
    transition_flag_UL=[]
    transition_flag_rssi=[]
    transition_flag_rsrq=[]
    transition_flag_rsrp=[]
    transition_flag_nrxsrq=[]
    transition_flag_nrxsrp=[]
    transition_flag_snr=[]
    transition_flag_CQI=[]
    transition_flag_speed=[]
    
    
    
    previous_category_DL = classify_DL(DL)
    previous_category_UL = classify_UL(UL)
    previous_category_rssi = classify_rssi(rssi)
    #print("previous_category_rssi",previous_category_rssi)
    previous_category_rsrq = classify_rsrq(rsrq)
    previous_category_rsrp = classify_rsrp(rsrp)
    previous_category_nrxsrq = classify_nrxsrq(nrxsrq)
    previous_category_nrxsrp = classify_nrxsrp(nrxsrp)
    previous_category_snr = classify_snr(snr)
    previous_category_CQI = classify_CQI(CQI)
    previous_category_speed = classify_speed(speed)

    
    
    
    for _ in range(total_seconds):
        DL = round(DL, 2)
        UL = round(UL, 2)
        rssi = round(rssi, 2)
        rsrq = round(rsrq, 2)
        rsrp = round(rsrp, 2)
        nrxsrp = round(nrxsrp, 2)
        nrxsrq = round(nrxsrq, 2)
        snr = round(snr, 2)
        CQI = round(CQI, 2)
        speed = round(speed, 2)
        
        
        
        DL_values.append(DL)
        UL_values.append(UL)
        rssi_values.append(rssi)
        rsrq_values.append(rsrq)
        rsrp_values.append(rsrp)
        nrxsrq_values.append(nrxsrq)
        nrxsrp_values.append(nrxsrp)
        snr_values.append(snr)
        CQI_values.append(CQI)
        speed_values.append(speed)

        # Select next category based on probabilities
        next_category_DL = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_DL == "Good" else
                    [25, 50, 25] if previous_category_DL == "Medium" else
                    [50, 25, 25]
        )[0]

        # Pick transition range and update DL
        DL_range = pick_transition_large(previous_category_DL, next_category_DL)
        DL_val = round(random.uniform(*DL_range), 2)
        DL += DL_val
        DL = round(np.clip(DL,5,50), 2)  # Keep DL within valid range

        # Classify new DL value
        current_category_DL = classify_DL(DL)

        # Check for handover transition (Good <-> Bad)
        transition_DL = 0
        if previous_category_DL is not None and current_category_DL is not None:
            if (previous_category_DL == "Good" and current_category_DL == "Bad") or \
               (previous_category_DL == "Bad" and current_category_DL == "Good"):
                transition_DL = 1

        transition_flag_DL.append(transition_DL)
        previous_category_DL = current_category_DL
        
        
        # Select next category based on probabilities
        next_category_UL = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_UL == "Good" else
                    [25, 50, 25] if previous_category_UL == "Medium" else
                    [50, 25, 25]
        )[0]

        
        UL_range = pick_transition_large(previous_category_UL, next_category_UL)
        UL_val = round(random.uniform(*UL_range), 2)
        UL += UL_val
        UL = round(np.clip(UL,1 , 15), 2)  
        current_category_UL = classify_UL(UL)
        # Check for handover transition (Good <-> Bad)
        transition_UL = 0
        if previous_category_UL is not None and current_category_UL is not None:
            if (previous_category_UL == "Good" and current_category_UL == "Bad") or \
               (previous_category_UL == "Bad" and current_category_UL == "Good"):
                transition_UL = 1

        transition_flag_UL.append(transition_UL)
        previous_category_UL = current_category_UL
        
        
        next_category_rssi = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_rssi == "Good" else
                    [25, 50, 25] if previous_category_rssi == "Medium" else
                    [50, 25, 25]
        )[0]
        rssi_range = pick_transition_large(previous_category_rssi, next_category_rssi)
        rssi_val = round(random.uniform(*rssi_range), 2)
        rssi += rssi_val
        rssi = round(np.clip(rssi, -100, -40), 2)
        current_category_rssi = classify_rssi(rssi)
        
        # Check for handover transition (Good <-> Bad)
        transition_rssi = 0
        if previous_category_rssi is not None and current_category_rssi is not None:
            if (previous_category_rssi == "Good" and current_category_rssi == "Bad") or \
               (previous_category_rssi == "Bad" and current_category_rssi == "Good"):
                transition_rssi = 1

        transition_flag_rssi.append(transition_rssi)
        
        previous_category_rssi = current_category_rssi
        
        
        next_category_rsrq = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_rsrq == "Good" else
                    [25, 50, 25] if previous_category_rsrq == "Medium" else
                    [50, 25, 25]
        )[0]
        rsrq_range = pick_transition_large(previous_category_rsrq, next_category_rsrq)
        rsrq_val = round(random.uniform(*rsrq_range), 2)
        rsrq += rsrq_val
        rsrq = round(np.clip(rsrq, -20, -3), 2)
        current_category_rsrq = classify_rsrq(rsrq)
        
        # Check for handover transition (Good <-> Bad)
        transition_rsrq = 0
        if previous_category_rsrq is not None and current_category_rsrq is not None:
            if (previous_category_rsrq == "Good" and current_category_rsrq == "Bad") or \
               (previous_category_rsrq == "Bad" and current_category_rsrq == "Good"):
                transition_rsrq = 1

        transition_flag_rsrq.append(transition_rsrq)
        
        previous_category_rsrq = current_category_rsrq
        
        
        next_category_rsrp = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_rsrp == "Good" else
                    [25, 50, 25] if previous_category_rsrp == "Medium" else
                    [50, 25, 25]
        )[0]
        rsrp_range = pick_transition_large(previous_category_rsrp, next_category_rsrp)
        rsrp_val = round(random.uniform(*rsrp_range), 2)
        rsrp += rsrp_val
        rsrp = round(np.clip(rsrp, -120, -30), 2)
        current_category_rsrp = classify_rsrp(rsrp)
        
        # Check for handover transition (Good <-> Bad)
        transition_rsrp = 0
        if previous_category_rsrp is not None and current_category_rsrp is not None:
            if (previous_category_rsrp == "Good" and current_category_rsrp == "Bad") or \
               (previous_category_rsrp == "Bad" and current_category_rsrp == "Good"):
                transition_rsrp = 1

        transition_flag_rsrp.append(transition_rsrp)
        
        previous_category_rsrp = current_category_rsrp
        
        
        next_category_nrxsrp = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_nrxsrp == "Good" else
                    [25, 50, 25] if previous_category_nrxsrp == "Medium" else
                    [50, 25, 25]
        )[0]
        nrxsrp_range = pick_transition_large(previous_category_nrxsrp, next_category_nrxsrp)
        nrxsrp_val = round(random.uniform(*nrxsrp_range), 2)
        nrxsrp += nrxsrp_val
        nrxsrp = round(np.clip(nrxsrp, -120, -30), 2)
        current_category_nrxsrp = classify_nrxsrp(nrxsrp)
        
        # Check for handover transition (Good <-> Bad)
        transition_nrxsrp = 0
        if previous_category_nrxsrp is not None and current_category_nrxsrp is not None:
            if (previous_category_nrxsrp == "Good" and current_category_nrxsrp == "Bad") or \
               (previous_category_nrxsrp == "Bad" and current_category_nrxsrp == "Good"):
                transition_nrxsrp = 1

        transition_flag_nrxsrp.append(transition_nrxsrp)
        
        previous_category_nrxsrp = current_category_nrxsrp
        
        
        next_category_nrxsrq = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_nrxsrq == "Good" else
                    [25, 50, 25] if previous_category_nrxsrq == "Medium" else
                    [50, 25, 25]
        )[0]
        nrxsrq_range = pick_transition_large(previous_category_nrxsrq, next_category_nrxsrq)
        nrxsrq_val = round(random.uniform(*nrxsrq_range), 2)
        nrxsrq += nrxsrq_val
        nrxsrq = round(np.clip(nrxsrq, -20, -3), 2)
        current_category_nrxsrq = classify_nrxsrq(nrxsrq)
        
        # Check for handover transition (Good <-> Bad)
        transition_nrxsrq = 0
        if previous_category_nrxsrq is not None and current_category_nrxsrq is not None:
            if (previous_category_nrxsrq == "Good" and current_category_nrxsrq == "Bad") or \
               (previous_category_nrxsrq == "Bad" and current_category_nrxsrq == "Good"):
                transition_nrxsrq = 1

        transition_flag_nrxsrq.append(transition_nrxsrq)
        previous_category_nrxsrq = current_category_nrxsrq
        
        
        
        
        next_category_snr = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_snr == "Good" else
                    [25, 50, 25] if previous_category_snr == "Medium" else
                    [50, 25, 25]
        )[0]
        snr_range = pick_transition_small(previous_category_snr, next_category_snr)
        snr_val = round(random.uniform(*snr_range), 2)
        snr += snr_val
        snr = round(np.clip(snr, 1, 20), 2)
        current_category_snr = classify_snr(snr)
        
        # Check for handover transition (Good <-> Bad)
        transition_snr = 0
        if previous_category_snr is not None and current_category_snr is not None:
            if (previous_category_snr == "Good" and current_category_snr == "Bad") or \
               (previous_category_snr == "Bad" and current_category_snr == "Good"):
                transition_snr = 1

        transition_flag_snr.append(transition_snr)
        
        previous_category_snr = current_category_snr
        
        
        next_category_CQI = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_CQI == "Good" else
                    [25, 50, 25] if previous_category_CQI == "Medium" else
                    [50, 25, 25]
        )[0]
#         print("next_category_CQI",next_category_CQI)
        CQI_range = pick_transition_large(previous_category_CQI, next_category_CQI)
#         print("CQI_range",CQI_range)
        CQI_val = round(random.uniform(*CQI_range), 2)
#         print("CQI_val",CQI_val)
        CQI += CQI_val
#         print("CQI",CQI)
        
        CQI = round(np.clip(CQI, 0,15), 2)
#         print("after update CQI",CQI)
        current_category_CQI = classify_CQI(CQI)
#         print("current_category_CQI",current_category_CQI)
        
        # Check for handover transition (Good <-> Bad)
        transition_CQI = 0
        if previous_category_CQI is not None and current_category_CQI is not None:
            if (previous_category_CQI == "Good" and current_category_CQI == "Bad") or \
               (previous_category_CQI == "Bad" and current_category_CQI == "Good"):
                transition_CQI = 1

        transition_flag_CQI.append(transition_CQI)
        #print("transition_flag_CQI value",transition_flag_CQI)
        
        previous_category_CQI = current_category_CQI
        
        
        next_category_speed = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_speed == "Good" else
                    [25, 50, 25] if previous_category_speed == "Medium" else
                    [50, 25, 25]
        )[0]
        speed_range = pick_transition_small(previous_category_speed, next_category_speed)
        speed_val = round(random.uniform(*speed_range), 2)
        speed += speed_val
        speed = round(np.clip(speed, 5, 50), 2)
        current_category_speed = classify_speed(speed)
        
        # Check for handover transition (Good <-> Bad)
        transition_speed = 0
        if previous_category_speed is not None and current_category_speed is not None:
            if (previous_category_speed == "Good" and current_category_speed == "Bad") or \
               (previous_category_speed == "Bad" and current_category_speed == "Good"):
                transition_speed = 1

        transition_flag_speed.append(transition_speed)
        
        previous_category_speed = current_category_speed
        
         
    
    # Calculate handover values
    handover = calculate_handover(total_seconds,w_DL,w_UL, w_rssi,w_rsrq,w_rsrp,w_nrxsrp,w_nrxsrq,w_snr,w_CQI,w_speed,transition_flag_DL,transition_flag_UL,transition_flag_rssi,transition_flag_rsrq,transition_flag_rsrp,transition_flag_nrxsrp,transition_flag_nrxsrq,transition_flag_snr,transition_flag_CQI,transition_flag_speed)
    # Return generated data as DataFrame
    return pd.DataFrame({"DL": DL_values,"UL": UL_values,"RSSI": rssi_values,"rsrq":rsrq_values,"rsrp": rsrp_values,"nrxsrp": nrxsrp_values,"nrxsrq": nrxsrq_values,"snr": snr_values,"CQI": CQI_values,"speed": speed_values,"transition_flag_DL":transition_flag_DL,"transition_flag_UL":transition_flag_UL,"transition_flag_rssi":transition_flag_rssi,"transition_flag_rsrq":transition_flag_rsrq,"transition_flag_rsrp":transition_flag_rsrp,"transition_flag_nrxsrp":transition_flag_nrxsrp,"transition_flag_nrxsrq":transition_flag_nrxsrq,"transition_flag_snr":transition_flag_snr,"transition_flag_CQI":transition_flag_CQI,"transition_flag_speed":transition_flag_speed, "Handover": handover})


# In[759]:


# Generate data and save to CSV
df = generate_data(total_seconds=300*60*60)
df.to_csv("synthetic_handover_data.csv", index=False)
print("Data generated and saved to 'synthetic_handover_data.csv'.")


# In[760]:


import pandas as pd

# Load the dataset
file_name = "synthetic_handover_data.csv"
summary_file = "handover_summary.csv"

# Read the CSV file
df = pd.read_csv(file_name)

# Check if 'Handover' column exists
if 'Handover' not in df.columns:
    print("Error: 'Handover' column not found in the dataset.")
else:
    # Count occurrences of 0 and 1 in the 'Handover' column
    count_0 = (df['Handover'] == 0).sum()
    count_1 = (df['Handover'] == 1).sum()

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Handover_Value': [0, 1],
        'Count': [count_0, count_1]
    })

    # Save the summary to a new CSV file
    summary_df.to_csv(summary_file, index=False)

    # Print summary and success message
    print(f"Summary of 'Handover' column saved to '{summary_file}'.")
    print(summary_df)


# In[761]:


import pandas as pd

# Load the dataset
file_name = "synthetic_handover_data.csv"
output_file = "handover_data_sum.csv"

# Read the CSV file with extra stripping to clean any whitespace
df = pd.read_csv(file_name).apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# List of transition flag columns to sum
transition_columns = [
    'transition_flag_DL', 'transition_flag_UL', 'transition_flag_rssi',
    'transition_flag_rsrq', 'transition_flag_rsrp', 'transition_flag_nrxsrp',
    'transition_flag_nrxsrq', 'transition_flag_snr', 'transition_flag_CQI',
    'transition_flag_speed'
]

# Check the data types of the columns
print("Column Data Types:\n", df.dtypes)

# Convert transition flag columns to integers if necessary
df[transition_columns] = df[transition_columns].apply(pd.to_numeric, errors='coerce')

# Add a new column with the sum of transition flag values
df['sum_transition_flags'] = df[transition_columns].sum(axis=1)

# Count rows where sum_transition_flags equals specific values
count_1 = (df['sum_transition_flags'] == 1).sum()
count_2 = (df['sum_transition_flags'] == 2).sum()
count_3 = (df['sum_transition_flags'] == 3).sum()
count_4 = (df['sum_transition_flags'] == 4).sum()
count_5 = (df['sum_transition_flags'] == 5).sum()
count_6 = (df['sum_transition_flags'] == 6).sum()
count_7 = (df['sum_transition_flags'] == 7).sum()
count_8 = (df['sum_transition_flags'] == 8).sum()

# Save the updated dataset to a new CSV file
df.to_csv(output_file, index=False)

# Create a summary DataFrame to store the counts
summary_df = pd.DataFrame({
    'sum_value': [1, 2, 3, 4,5, 6, 7, 8],
    'count': [count_1, count_2, count_3, count_4,count_5, count_6, count_7, count_8]
})

# Save the summary to a new CSV file
summary_file = "handover_summary.csv"
summary_df.to_csv(summary_file, index=False)

# Print summary
print(f"File saved successfully as '{output_file}' with 'sum_transition_flags' column.")
print(f"Summary file saved as '{summary_file}' with counts of specific values.")
print(summary_df)


# In[778]:


# Generate synthetic DL data and handover status
def generate_data(total_seconds, w_DL=0.12,w_UL=0.12, w_rssi=0.12,w_rsrq=0.12,w_rsrp=0.12,w_nrxsrp=0.12,w_nrxsrq=0.12,w_snr=0.12,w_CQI=0.47,w_speed=0.12):
    np.random.seed(None)
    random.seed(None)

    
    DL = random.uniform(5,50)
    UL = random.uniform(1,15)
    rssi = random.uniform(-100, -40)
    rsrq = random.uniform(-20, -3)
    rsrp = random.uniform(-120, -30)
    nrxsrp=random.uniform(-120, -30)
    nrxsrq=random.uniform(-20, -3)
    snr=random.uniform(1,20)
    CQI=random.uniform(0,15)
    speed=random.uniform(5,50)
    
    
    
    
    DL_values = []
    UL_values = []
    rssi_values = []
    rsrq_values = []
    rsrp_values = []
    nrxsrp_values = []
    nrxsrq_values = []
    snr_values = []
    CQI_values = []
    speed_values = []
    
    
    
    
    transition_flag_DL=[]
    transition_flag_UL=[]
    transition_flag_rssi=[]
    transition_flag_rsrq=[]
    transition_flag_rsrp=[]
    transition_flag_nrxsrq=[]
    transition_flag_nrxsrp=[]
    transition_flag_snr=[]
    transition_flag_CQI=[]
    transition_flag_speed=[]
    
    
    
    previous_category_DL = classify_DL(DL)
    previous_category_UL = classify_UL(UL)
    previous_category_rssi = classify_rssi(rssi)
    #print("previous_category_rssi",previous_category_rssi)
    previous_category_rsrq = classify_rsrq(rsrq)
    previous_category_rsrp = classify_rsrp(rsrp)
    previous_category_nrxsrq = classify_nrxsrq(nrxsrq)
    previous_category_nrxsrp = classify_nrxsrp(nrxsrp)
    previous_category_snr = classify_snr(snr)
    previous_category_CQI = classify_CQI(CQI)
    previous_category_speed = classify_speed(speed)

    
    
    
    for _ in range(total_seconds):
        DL = round(DL, 2)
        UL = round(UL, 2)
        rssi = round(rssi, 2)
        rsrq = round(rsrq, 2)
        rsrp = round(rsrp, 2)
        nrxsrp = round(nrxsrp, 2)
        nrxsrq = round(nrxsrq, 2)
        snr = round(snr, 2)
        CQI = round(CQI, 2)
        speed = round(speed, 2)
        
        
        
        DL_values.append(DL)
        UL_values.append(UL)
        rssi_values.append(rssi)
        rsrq_values.append(rsrq)
        rsrp_values.append(rsrp)
        nrxsrq_values.append(nrxsrq)
        nrxsrp_values.append(nrxsrp)
        snr_values.append(snr)
        CQI_values.append(CQI)
        speed_values.append(speed)

        # Select next category based on probabilities
        next_category_DL = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_DL == "Good" else
                    [25, 50, 25] if previous_category_DL == "Medium" else
                    [50, 25, 25]
        )[0]

        # Pick transition range and update DL
        DL_range = pick_transition_large(previous_category_DL, next_category_DL)
        DL_val = round(random.uniform(*DL_range), 2)
        DL += DL_val
        DL = round(np.clip(DL,5,50), 2)  # Keep DL within valid range

        # Classify new DL value
        current_category_DL = classify_DL(DL)

        # Check for handover transition (Good <-> Bad)
        transition_DL = 0
        if previous_category_DL is not None and current_category_DL is not None:
            if (previous_category_DL == "Good" and current_category_DL == "Bad") or \
               (previous_category_DL == "Bad" and current_category_DL == "Good"):
                transition_DL = 1

        transition_flag_DL.append(transition_DL)
        previous_category_DL = current_category_DL
        
        
        # Select next category based on probabilities
        next_category_UL = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_UL == "Good" else
                    [25, 50, 25] if previous_category_UL == "Medium" else
                    [50, 25, 25]
        )[0]

        
        UL_range = pick_transition_large(previous_category_UL, next_category_UL)
        UL_val = round(random.uniform(*UL_range), 2)
        UL += UL_val
        UL = round(np.clip(UL,1 , 15), 2)  
        current_category_UL = classify_UL(UL)
        # Check for handover transition (Good <-> Bad)
        transition_UL = 0
        if previous_category_UL is not None and current_category_UL is not None:
            if (previous_category_UL == "Good" and current_category_UL == "Bad") or \
               (previous_category_UL == "Bad" and current_category_UL == "Good"):
                transition_UL = 1

        transition_flag_UL.append(transition_UL)
        previous_category_UL = current_category_UL
        
        
        next_category_rssi = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_rssi == "Good" else
                    [25, 50, 25] if previous_category_rssi == "Medium" else
                    [50, 25, 25]
        )[0]
        rssi_range = pick_transition_large(previous_category_rssi, next_category_rssi)
        rssi_val = round(random.uniform(*rssi_range), 2)
        rssi += rssi_val
        rssi = round(np.clip(rssi, -100, -40), 2)
        current_category_rssi = classify_rssi(rssi)
        
        # Check for handover transition (Good <-> Bad)
        transition_rssi = 0
        if previous_category_rssi is not None and current_category_rssi is not None:
            if (previous_category_rssi == "Good" and current_category_rssi == "Bad") or \
               (previous_category_rssi == "Bad" and current_category_rssi == "Good"):
                transition_rssi = 1

        transition_flag_rssi.append(transition_rssi)
        
        previous_category_rssi = current_category_rssi
        
        
        next_category_rsrq = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_rsrq == "Good" else
                    [25, 50, 25] if previous_category_rsrq == "Medium" else
                    [50, 25, 25]
        )[0]
        rsrq_range = pick_transition_large(previous_category_rsrq, next_category_rsrq)
        rsrq_val = round(random.uniform(*rsrq_range), 2)
        rsrq += rsrq_val
        rsrq = round(np.clip(rsrq, -20, -3), 2)
        current_category_rsrq = classify_rsrq(rsrq)
        
        # Check for handover transition (Good <-> Bad)
        transition_rsrq = 0
        if previous_category_rsrq is not None and current_category_rsrq is not None:
            if (previous_category_rsrq == "Good" and current_category_rsrq == "Bad") or \
               (previous_category_rsrq == "Bad" and current_category_rsrq == "Good"):
                transition_rsrq = 1

        transition_flag_rsrq.append(transition_rsrq)
        
        previous_category_rsrq = current_category_rsrq
        
        
        next_category_rsrp = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_rsrp == "Good" else
                    [25, 50, 25] if previous_category_rsrp == "Medium" else
                    [50, 25, 25]
        )[0]
        rsrp_range = pick_transition_large(previous_category_rsrp, next_category_rsrp)
        rsrp_val = round(random.uniform(*rsrp_range), 2)
        rsrp += rsrp_val
        rsrp = round(np.clip(rsrp, -120, -30), 2)
        current_category_rsrp = classify_rsrp(rsrp)
        
        # Check for handover transition (Good <-> Bad)
        transition_rsrp = 0
        if previous_category_rsrp is not None and current_category_rsrp is not None:
            if (previous_category_rsrp == "Good" and current_category_rsrp == "Bad") or \
               (previous_category_rsrp == "Bad" and current_category_rsrp == "Good"):
                transition_rsrp = 1

        transition_flag_rsrp.append(transition_rsrp)
        
        previous_category_rsrp = current_category_rsrp
        
        
        next_category_nrxsrp = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_nrxsrp == "Good" else
                    [25, 50, 25] if previous_category_nrxsrp == "Medium" else
                    [50, 25, 25]
        )[0]
        nrxsrp_range = pick_transition_large(previous_category_nrxsrp, next_category_nrxsrp)
        nrxsrp_val = round(random.uniform(*nrxsrp_range), 2)
        nrxsrp += nrxsrp_val
        nrxsrp = round(np.clip(nrxsrp, -120, -30), 2)
        current_category_nrxsrp = classify_nrxsrp(nrxsrp)
        
        # Check for handover transition (Good <-> Bad)
        transition_nrxsrp = 0
        if previous_category_nrxsrp is not None and current_category_nrxsrp is not None:
            if (previous_category_nrxsrp == "Good" and current_category_nrxsrp == "Bad") or \
               (previous_category_nrxsrp == "Bad" and current_category_nrxsrp == "Good"):
                transition_nrxsrp = 1

        transition_flag_nrxsrp.append(transition_nrxsrp)
        
        previous_category_nrxsrp = current_category_nrxsrp
        
        
        next_category_nrxsrq = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_nrxsrq == "Good" else
                    [25, 50, 25] if previous_category_nrxsrq == "Medium" else
                    [50, 25, 25]
        )[0]
        nrxsrq_range = pick_transition_large(previous_category_nrxsrq, next_category_nrxsrq)
        nrxsrq_val = round(random.uniform(*nrxsrq_range), 2)
        nrxsrq += nrxsrq_val
        nrxsrq = round(np.clip(nrxsrq, -20, -3), 2)
        current_category_nrxsrq = classify_nrxsrq(nrxsrq)
        
        # Check for handover transition (Good <-> Bad)
        transition_nrxsrq = 0
        if previous_category_nrxsrq is not None and current_category_nrxsrq is not None:
            if (previous_category_nrxsrq == "Good" and current_category_nrxsrq == "Bad") or \
               (previous_category_nrxsrq == "Bad" and current_category_nrxsrq == "Good"):
                transition_nrxsrq = 1

        transition_flag_nrxsrq.append(transition_nrxsrq)
        previous_category_nrxsrq = current_category_nrxsrq
        
        
        
        
        next_category_snr = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_snr == "Good" else
                    [25, 50, 25] if previous_category_snr == "Medium" else
                    [50, 25, 25]
        )[0]
        snr_range = pick_transition_small(previous_category_snr, next_category_snr)
        snr_val = round(random.uniform(*snr_range), 2)
        snr += snr_val
        snr = round(np.clip(snr, 1, 20), 2)
        current_category_snr = classify_snr(snr)
        
        # Check for handover transition (Good <-> Bad)
        transition_snr = 0
        if previous_category_snr is not None and current_category_snr is not None:
            if (previous_category_snr == "Good" and current_category_snr == "Bad") or \
               (previous_category_snr == "Bad" and current_category_snr == "Good"):
                transition_snr = 1

        transition_flag_snr.append(transition_snr)
        
        previous_category_snr = current_category_snr
        
        
        next_category_CQI = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_CQI == "Good" else
                    [25, 50, 25] if previous_category_CQI == "Medium" else
                    [50, 25, 25]
        )[0]
#         print("next_category_CQI",next_category_CQI)
        CQI_range = pick_transition_large(previous_category_CQI, next_category_CQI)
#         print("CQI_range",CQI_range)
        CQI_val = round(random.uniform(*CQI_range), 2)
#         print("CQI_val",CQI_val)
        CQI += CQI_val
#         print("CQI",CQI)
        
        CQI = round(np.clip(CQI, 0,15), 2)
#         print("after update CQI",CQI)
        current_category_CQI = classify_CQI(CQI)
#         print("current_category_CQI",current_category_CQI)
        
        # Check for handover transition (Good <-> Bad)
        transition_CQI = 0
        if previous_category_CQI is not None and current_category_CQI is not None:
            if (previous_category_CQI == "Good" and current_category_CQI == "Bad") or \
               (previous_category_CQI == "Bad" and current_category_CQI == "Good"):
                transition_CQI = 1

        transition_flag_CQI.append(transition_CQI)
        #print("transition_flag_CQI value",transition_flag_CQI)
        
        previous_category_CQI = current_category_CQI
        
        
        next_category_speed = random.choices(
            ["Good", "Medium", "Bad"],
            weights=[50, 10, 40] if previous_category_speed == "Good" else
                    [25, 50, 25] if previous_category_speed == "Medium" else
                    [50, 25, 25]
        )[0]
        speed_range = pick_transition_small(previous_category_speed, next_category_speed)
        speed_val = round(random.uniform(*speed_range), 2)
        speed += speed_val
        speed = round(np.clip(speed, 5, 50), 2)
        current_category_speed = classify_speed(speed)
        
        # Check for handover transition (Good <-> Bad)
        transition_speed = 0
        if previous_category_speed is not None and current_category_speed is not None:
            if (previous_category_speed == "Good" and current_category_speed == "Bad") or \
               (previous_category_speed == "Bad" and current_category_speed == "Good"):
                transition_speed = 1

        transition_flag_speed.append(transition_speed)
        
        previous_category_speed = current_category_speed
        
         
    
    # Calculate handover values
    handover = calculate_handover(total_seconds,w_DL,w_UL, w_rssi,w_rsrq,w_rsrp,w_nrxsrp,w_nrxsrq,w_snr,w_CQI,w_speed,transition_flag_DL,transition_flag_UL,transition_flag_rssi,transition_flag_rsrq,transition_flag_rsrp,transition_flag_nrxsrp,transition_flag_nrxsrq,transition_flag_snr,transition_flag_CQI,transition_flag_speed)
    # Return generated data as DataFrame
    return pd.DataFrame({"DL": DL_values,"UL": UL_values,"RSSI": rssi_values,"rsrq":rsrq_values,"rsrp": rsrp_values,"nrxsrp": nrxsrp_values,"nrxsrq": nrxsrq_values,"snr": snr_values,"CQI": CQI_values,"speed": speed_values,"transition_flag_DL":transition_flag_DL,"transition_flag_UL":transition_flag_UL,"transition_flag_rssi":transition_flag_rssi,"transition_flag_rsrq":transition_flag_rsrq,"transition_flag_rsrp":transition_flag_rsrp,"transition_flag_nrxsrp":transition_flag_nrxsrp,"transition_flag_nrxsrq":transition_flag_nrxsrq,"transition_flag_snr":transition_flag_snr,"transition_flag_CQI":transition_flag_CQI,"transition_flag_speed":transition_flag_speed, "Handover": handover})


# In[781]:


# Generate data and save to CSV
df = generate_data(total_seconds=300*60*60)
df.to_csv("synthetic_handover_data_1.csv", index=False)
print("Data generated and saved to 'synthetic_handover_data_1.csv'.")


# In[782]:


import pandas as pd

# Load the dataset
file_name = "synthetic_handover_data_1.csv"
summary_file = "handover_summary.csv"

# Read the CSV file
df = pd.read_csv(file_name)

# Check if 'Handover' column exists
if 'Handover' not in df.columns:
    print("Error: 'Handover' column not found in the dataset.")
else:
    # Count occurrences of 0 and 1 in the 'Handover' column
    count_0 = (df['Handover'] == 0).sum()
    count_1 = (df['Handover'] == 1).sum()

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Handover_Value': [0, 1],
        'Count': [count_0, count_1]
    })

    # Save the summary to a new CSV file
    summary_df.to_csv(summary_file, index=False)

    # Print summary and success message
    print(f"Summary of 'Handover' column saved to '{summary_file}'.")
    print(summary_df)


# In[ ]:




