
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"data"/"calories.csv"
OUT = ROOT/"outputs"
MODELS = ROOT/"models"
st.set_page_config(page_title="Calories Burnt Analytics", page_icon="🔥", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0b1020 0%,#111827 55%,#17152b 100%); color:#f3f4f6;}
[data-testid="stSidebar"] {
    background: #304568;
    color: white;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: white !important;
}
/* EDA tab text color */
.stApp button[data-baseweb="tab"] {
    color: white !important;
}

.stApp button[data-baseweb="tab"] p {
    color: white !important;
}

/* Selected tab */
.stApp button[data-baseweb="tab"][aria-selected="true"] {
    color: white !important;
}
[data-testid="stMetric"] {
    background: #40577A;
    border: 1px solid #60799D;
    padding: 16px;
    border-radius: 14px;
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: white !important;
}
h1,h2,h3 {color:#F54927;}
/* EDA tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #40577A;
    border-radius: 10px;
    padding: 6px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #40577A;
    color: white !important;
}

.stTabs [data-baseweb="tab"] p {
    color: white !important;
}

.stTabs [aria-selected="true"] {
    background-color: #60799D !important;
}


/* EDA numeric variable dropdown */
.stApp label {
    color: white !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #40577A !important;
    color: white !important;
    border-color: #60799D !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: white !important;
}

[data-baseweb="popover"] ul {
    background-color: #40577A !important;
}

[data-baseweb="popover"] li {
    background-color: #40577A !important;
    color: white !important;
}

[data-baseweb="popover"] li:hover {
    background-color: #60799D !important;
    color: white !important;
}


</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA)
@st.cache_data
def load_metrics():
    p=OUT/"model_comparison.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
@st.cache_resource
def load_model(name):
    return joblib.load(MODELS/f"{name.lower().replace(' ','_')}.joblib")

st.title("🔥 Calories Burnt Prediction")
st.caption("Interactive analytics dashboard • Exploratory analysis • Four regression models")
df=load_data()
metrics=load_metrics()
with st.sidebar:
    st.header("Dashboard Controls")
    page=st.radio("Navigate",["Overview","Exploratory Data Analysis","Model Performance","Make a Prediction","Data Quality"])
    st.markdown("---")
    st.caption("Dataset: calories.csv")
    st.caption(f"{len(df):,} records")

if page=="Overview":
    st.subheader("Dataset at a glance")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Records",f"{len(df):,}")
    c2.metric("Features",str(df.shape[1]-1))
    c3.metric("Average calories",f"{df.Calories.mean():.1f}")
    c4.metric("Average duration",f"{df.Duration.mean():.1f} min")
    a,b=st.columns(2)
    with a: st.plotly_chart(px.histogram(df,x="Calories",nbins=35,title="Calories distribution",template="plotly_white"),use_container_width=True)
    with b: st.plotly_chart(px.scatter(df.sample(min(2500,len(df)),random_state=42),x="Duration",y="Calories",color="Gender",title="Duration vs calories",template="plotly_dark"),use_container_width=True)
    st.dataframe(df.head(10),use_container_width=True)
elif page=="Exploratory Data Analysis":
    st.subheader("Univariate, bivariate and multivariate analysis")
    tab1,tab2,tab3=st.tabs(["Univariate","Bivariate","Multivariate"])
    with tab1:
        col=st.selectbox("Choose a numeric variable",df.select_dtypes(include=np.number).columns)
        st.plotly_chart(px.histogram(df,x=col,marginal="box",template="plotly_dark",title=f"Distribution of {col}"),use_container_width=True)
    with tab2:
        x=st.selectbox("X-axis",["Duration","Age","Weight","Height","Heart_Rate","Body_Temp"],index=0)
        y=st.selectbox("Y-axis",["Calories","Duration","Heart_Rate","Body_Temp"],index=0)
        st.plotly_chart(px.scatter(df.sample(min(4000,len(df)),random_state=42),x=x,y=y,color="Gender",template="plotly_dark"),use_container_width=True)
        st.plotly_chart(px.box(df,x="Gender",y="Calories",template="plotly_dark"),use_container_width=True)
    with tab3:
        corr=df.select_dtypes(include=np.number).corr()
        st.plotly_chart(px.imshow(corr,text_auto=".2f",aspect="auto",template="plotly_dark",title="Correlation heatmap"),use_container_width=True)
        dims=["Age","Weight","Duration","Heart_Rate","Body_Temp","Calories"]
        st.plotly_chart(px.scatter_matrix(df.sample(min(1000,len(df)),random_state=42),dimensions=dims,color="Gender",template="plotly_dark"),use_container_width=True)
elif page=="Model Performance":
    st.subheader("Regression model comparison")
    if metrics.empty:
        st.warning("Run `python train_model.py` first to train models.")
    else:
        st.dataframe(metrics[["Model","MAE","RMSE","R2","Best_CV_MAE"]].style.format(precision=3),use_container_width=True)
        st.plotly_chart(px.bar(metrics,x="Model",y="RMSE",color="Model",template="plotly_dark",title="RMSE comparison (lower is better)"),use_container_width=True)
        st.plotly_chart(px.bar(metrics,x="Model",y="R2",color="Model",template="plotly_dark",title="R² comparison (higher is better)"),use_container_width=True)
        best=json.loads((OUT/"best_model.json").read_text())["best_model"]
        st.success(f"Best model by test RMSE: {best}")
elif page=="Make a Prediction":
    st.subheader("Estimate calories burnt")
    c1,c2,c3=st.columns(3)
    with c1:
        gender=st.selectbox("Gender",["female","male"])
        age=st.number_input("Age",min_value=1,max_value=100,value=25)
        height=st.number_input("Height (cm)",min_value=80.0,max_value=230.0,value=170.0)
    with c2:
        weight=st.number_input("Weight (kg)",min_value=20.0,max_value=250.0,value=65.0)
        duration=st.number_input("Duration (minutes)",min_value=1.0,max_value=300.0,value=30.0)
        heart=st.number_input("Heart rate (bpm)",min_value=40.0,max_value=220.0,value=100.0)
    with c3:
        temp=st.number_input("Body temperature (°C)",min_value=30.0,max_value=43.0,value=38.0)
        model_name=st.selectbox("Model",["Linear Regression","KNN","Decision Tree","Random Forest"])
    if st.button("Predict calories",type="primary"):
        model=load_model(model_name)
        row=pd.DataFrame([{"Gender":gender,"Age":age,"Height":height,"Weight":weight,"Duration":duration,"Heart_Rate":heart,"Body_Temp":temp}])
        prediction=float(model.predict(row)[0])
        st.metric("Estimated calories burnt",f"{prediction:.1f} kcal")
        st.caption("This is a model estimate based on the dataset, not a medical or fitness measurement.")
elif page=="Data Quality":
    st.subheader("Cleaning and quality checks")
    qpath=OUT/"data_quality.json"
    if qpath.exists():
        q=json.loads(qpath.read_text())
        a,b,c=st.columns(3)
        a.metric("Missing values (initial)",sum(q["missing_values_before"].values()))
        b.metric("Duplicate rows (initial)",q["duplicate_rows_before"])
        c.metric("Rows after cleaning",q["rows_after_cleaning"])
        st.write("Missing-value counts")
        st.json(q["missing_values_before"])
        st.write("IQR outlier inspection (flagged values are retained)")
        st.dataframe(pd.DataFrame(q["iqr_outlier_inspection"]).T,use_container_width=True)
    else:
        st.info("Run training to generate the data-quality report.")
    st.write("Dataset preview")
    st.dataframe(df.describe(include="all").T,use_container_width=True)
