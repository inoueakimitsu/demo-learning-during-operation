import numpy as np

from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt
import matplotlib
plt.style.use('ggplot')
font = {'family': 'meiryo'}
matplotlib.rc('font', **font)

import streamlit as st
from streamlit_autorefresh import st_autorefresh

def true_data(t):
    return np.sin(t/5) + np.random.normal(loc=0, scale=0.001, size=len(t) if not isinstance(t, int) else 1)
    # return np.sin(t/7 + np.cos(t/3)) + np.random.normal(loc=0, scale=0.001, size=len(t) if not isinstance(t, int) else 1)

if 'ts' not in st.session_state or 'ys' not in st.session_state:
    st.session_state['ts'] = np.arange(-10, 0)
    st.session_state['ys'] = true_data(st.session_state['ts']).tolist()
    st.session_state['ts'] = st.session_state['ts'].tolist()

st.title('Demo: Learning during operation')

'''This is a demonstration of a time series forecasting model that continuously learns while in operation.'''

'''Learning is performed approximately once per second.
The learning process uses the K nearest neighbor method, which has a small computational load.'''

new_t = st_autorefresh(interval=1000, limit=10000, key="data_input")
new_y = true_data(new_t)[0]

st.session_state['ts'].append(new_t)
st.session_state['ys'].append(new_y)

window = 3
horizon = 20

X = []
Y = []
for i in range(len(st.session_state['ts'])-window-horizon):
    X.append(st.session_state['ys'][i:i+window])
    Y.append(st.session_state['ys'][i+window+1])
X = np.array(X)
Y = np.array(Y)

model = KNeighborsRegressor(n_neighbors=1)
model.fit(X, Y)

pred_ys = st.session_state['ys'][-window:]
pred_ts = st.session_state['ts'][-window:]
for i in range(horizon):
    pred_ys.append(model.predict(
        [pred_ys[-window:]]
    )[0])
    pred_ts.append(new_t + i)

fig = plt.figure(figsize=(12, 4))
ax = plt.axes()
plt.plot(st.session_state['ts'], st.session_state['ys'], label="Observed")
plt.plot(pred_ts, pred_ys, "--", label="Predicted")
plt.ylabel("value")
plt.xlabel("time")
plt.xlim(left=new_t-100, right=new_t+horizon+window)
plt.legend()

st.pyplot(fig)
