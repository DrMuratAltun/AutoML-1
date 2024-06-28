#!/usr/bin/env python
# coding: utf-8

# # MLOPS- Pycaret Kullanımı

# In[2]:


#pip install ydata_profiling


# In[3]:


#pip install streamlit_pandas_profiling


# In[4]:


import streamlit as st
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from streamlit_pandas_profiling import st_profile_report
import ydata_profiling
import os


# In[ ]:


with st.sidebar:
    st.image('https://leilaabdel.com/img/deep_learning_course_pic.png')
    st.title('AutoML Classification')
    choice=st.radio('Navigaiton',['Upload','EDA','Modelling','Download'])
    

