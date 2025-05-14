import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv("concrete.csv")

# Feature engineering
df['water_cement_ratio'] = df['water'] / df['cement']
df['total_binder'] = df['cement'] + df['slag'] + df['ash']
df['water_per_binder'] = df['water'] / df['total_binder']
df['cement_share'] = df['cement'] / df['total_binder']
df['cement_x_sp'] = df['cement'] * df['superplastic']
df['slag_x_water'] = df['slag'] * df['water']
df['log_age'] = np.log1p(df['age'])
df['sp_water_ratio'] = df['superplastic'] / df['water']
df['agg_ratio'] = df['coarseagg'] / df['fineagg']
df['sp_per_binder'] = df['superplastic'] / df['total_binder']
df['total_mass'] = (
    df['cement'] + df['slag'] + df['ash'] + df['water'] +
    df['superplastic'] + df['coarseagg'] + df['fineagg']
)

# Train model
X = df.drop(columns=['strength'])
y = df['strength']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05)
model.fit(X_train, y_train)

# App Layout
st.set_page_config(page_title="Concrete Strength Predictor", layout="wide", page_icon='👷🏻‍♂️')

st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["🧠 ML", "📊 Data Analysis", "ℹ️ About App", "👨‍💻 About Me"])

if tab == "🧠 ML":
    st.title("👨‍💻 Concrete Strength Predictor")

    st.markdown("Input mix details below to predict compressive strength:")

    cement = st.number_input("Cement (kg/m³)", 100, 600)
    slag = st.number_input("Slag (kg/m³)", 0, 300)
    ash = st.number_input("Fly Ash (kg/m³)", 0, 300)
    water = st.number_input("Water (kg/m³)", 100, 250)
    superplastic = st.number_input("Superplasticizer (kg/m³)", 0.0, 30.0)
    coarseagg = st.number_input("Coarse Aggregate (kg/m³)", 800, 1200)
    fineagg = st.number_input("Fine Aggregate (kg/m³)", 500, 1000)
    age = st.number_input("Age (days)", 1, 365)

    if st.button("Predict Strength"):
        total_binder = cement + slag + ash
        input_data = pd.DataFrame([{
            'cement': cement,
            'slag': slag,
            'ash': ash,
            'water': water,
            'superplastic': superplastic,
            'coarseagg': coarseagg,
            'fineagg': fineagg,
            'age': age,
            'water_cement_ratio': water / cement,
            'total_binder': total_binder,
            'water_per_binder': water / total_binder,
            'cement_share': cement / total_binder,
            'cement_x_sp': cement * superplastic,
            'slag_x_water': slag * water,
            'log_age': np.log1p(age),
            'sp_water_ratio': superplastic / water,
            'agg_ratio': coarseagg / fineagg,
            'sp_per_binder': superplastic / total_binder,
            'total_mass': cement + slag + ash + water + superplastic + coarseagg + fineagg
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted Compressive Strength: **{prediction:.2f} MPa**")

elif tab == "📊 Data Analysis":
    plt.grid(True)

    st.title("📊 Data Analysis")
    st.markdown("Explore various visualizations and patterns from the dataset.")

    # Correlation Heatmap
    st.subheader("🔗 Correlation Heatmap")
    st.markdown("Shows the linear correlation between all numerical features using Pearson coefficient.")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    # Age vs Strength
    st.subheader("📈 Age vs Strength Line Plot")
    st.markdown("Displays how compressive strength increases with age, following the concrete curing pattern.")
    sns.lineplot(x='age', y='strength', data=df.sort_values('age'))
    st.pyplot(plt.gcf())
    plt.clf()

    # Water-Binder Ratio
    st.subheader("💧 Water-Binder Ratio vs Strength (Colored by Age)")
    st.markdown("""
    - **Purpose:** Visualize how the water-to-binder ratio affects compressive strength.
    - **Insight:** Typically, a lower water-to-binder ratio results in higher strength, and older samples are generally stronger.
    """)
    df['water_binder_ratio'] = df['water'] / (df['cement'] + df['slag'] + df['ash'])
    sns.scatterplot(data=df, x='water_binder_ratio', y='strength', hue='age', palette='coolwarm')
    st.pyplot(plt.gcf())
    plt.clf()

    # Actual vs Predicted
    st.subheader("📊 Actual vs Predicted Strength")
    st.markdown("Shows model prediction accuracy compared to actual values.")
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.7, color='dodgerblue', label='Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             color='darkorange', linestyle='--', linewidth=2, label='Perfect Prediction Line (y = x)')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.title('Actual vs Predicted Concrete Strength')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(plt.gcf())
    plt.clf()

    # Violin Plot
    st.subheader("🎻 Violin Plot: Strength Distribution at Different Ages")
    st.markdown("""
    - **Purpose:** Combines boxplot and kernel density estimation to show strength distribution across ages.
    - **Insight:** Strength tends to increase with age and stabilize later, aligning with concrete hydration theory.
    """)
    sns.violinplot(data=df, x='age', y='strength', inner='quartile', scale='width')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()

    # 3D Plot: Cement vs Water vs Strength
    st.subheader("🧱 3D Plot: Cement vs Water vs Strength")
    st.markdown("""
    - **Purpose:** Visualizes how two main ingredients (cement and water) affect strength in 3D.
    - **Insight:** Higher cement and lower water typically result in greater strength.
    """)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['cement'], df['water'], df['strength'], c=df['strength'], cmap='viridis')
    ax.set_xlabel('Cement')
    ax.set_ylabel('Water')
    ax.set_zlabel('Strength')
    st.pyplot(fig)

    # PCA Projection
    st.subheader("🧬 PCA Projection (2D) Colored by Strength")
    st.markdown("""
    - **Purpose:** Reduces multiple features into 2D while preserving variance.
    - **Insight:** Reveals that similar strength samples group together, validating the feature selection.
    """)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    features = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']
    X_scaled = StandardScaler().fit_transform(df[features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['strength'], cmap='plasma', alpha=0.7)
    plt.colorbar(label='Strength')
    plt.title('PCA Projection (2D) Colored by Strength')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    st.pyplot(plt.gcf())
    plt.clf()

elif tab == "ℹ️ About App":
    st.title("ℹ️ About This App")
    st.markdown("""
    This web application predicts the **compressive strength** of concrete using machine learning.
    
    ### 🔍 Features:
    - Inputs for mix design components
    - Real-time prediction using **XGBoost model**
    - Visual analysis of trends
    - Engineering-based feature enhancements

    **Target Variable:** `Strength (MPa)`  
    **Tools Used:** Python, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit

    Dataset: Preprocessed version of the **Concrete Compressive Strength Dataset** (SI Units)
    """)

elif tab == "👨‍💻 About Me":
    st.title("👨‍💻 About the Creator")

    st.markdown("""
    **Name:** S.M. Abdullah Abdulbadeeii  
    **Profession:** Student | Engineer | Visionary  
    **Company:** SMAASU Corporation  
    **Goal:** Bringing societal change through **spirituality, logic, technology, and perpetuity**.

    🧠 Believer in learning by doing.  
    📌 Passionate about **AI**, **ML**, **engineering**, and the **betterment of Ummah**.

    ✨ *"Leaving a lasting essence in the world is more important than just existing in it."*
    """)

