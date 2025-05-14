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
tab = st.sidebar.radio("Go to", ["🧠 ML", "📊 Data Analysis", "📕 About App", "🎓 About Us", "👨‍💻 About Me"])

if tab == "🧠 ML":
    st.title("👨‍💻 Concrete Strength Predictor")

    st.markdown("Input mix details below to predict compressive strength:")

    cement = st.number_input("Cement (kg/m³)")
    slag = st.number_input("Slag (kg/m³)")
    ash = st.number_input("Fly Ash (kg/m³)")
    water = st.number_input("Water (kg/m³)")
    superplastic = st.number_input("Superplasticizer (kg/m³)")
    coarseagg = st.number_input("Coarse Aggregate (kg/m³)")
    fineagg = st.number_input("Fine Aggregate (kg/m³)")
    age = st.number_input("Age (days)")

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
    plt.grid(True, linestyle='--', alpha=0.5)
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
    plt.grid(True, linestyle='--', alpha=0.5)
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

elif tab == "📕 About App":
    st.title("About This Application")
    st.image("https://blog.novatr.com/hubfs/A%20robot%20examining%20the%20architectural%20plan.png",
             use_container_width=True, caption="Concrete Strength Meets Machine Learning")

    st.markdown("""
    <h3 style='color:#4A90E2;'>🏗️ What is This App?</h3>
    <p>
        This app predicts the <strong>compressive strength</strong> of concrete using advanced machine learning models based on the ingredient mix proportions.
        It is designed for engineers, data scientists, and educators to simulate or validate concrete mix performance without the need for lab testing.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3 style='color:#4A90E2;'>🧠 Technology Stack</h3>
    <ul>
        <li><strong>Python 3.10</strong></li>
        <li><strong>Scikit-learn, XGBoost</strong> for ML</li>
        <li><strong>Pandas, NumPy</strong> for data handling</li>
        <li><strong>Matplotlib, Seaborn</strong> for visualizations</li>
        <li><strong>Streamlit</strong> for the web interface</li>
    </ul>
    """, unsafe_allow_html=True)

    st.image("https://cdn.prod.website-files.com/60d07e2eecb304cb4350b53f/66545cdc3d33685d1f5ea0b0_ai%20in%20civil%20engineering%20-%20cover.jpg",
             use_container_width=True, caption="Machine Learning Pipeline (Source: GitHub - dipanjanS)")

    st.markdown("""
    <h3 style='color:#4A90E2;'>📊 Key Features</h3>
    <ul>
        <li>🎯 Predicts concrete strength from user-defined mix proportions</li>
        <li>🔍 Interactive data visualizations and statistical insights</li>
        <li>🧪 Uses engineered features like water-cement ratio, binder weight, log-age etc.</li>
        <li>🤖 Trained on real-world dataset with XGBoost and other ML models</li>
        <li>📥 Uses data from UCI Machine Learning Repository (SI units)</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3 style='color:#4A90E2;'>🏗️ Why This Matters</h3>
    <p>
        Concrete strength prediction is vital for infrastructure planning, cost optimization, and quality control. 
        Traditional methods are time-consuming and costly. This tool enables early-stage design decisions using machine learning.
    </p>
    """, unsafe_allow_html=True)

    st.image("https://www.batchmix.co.uk/wp-content/uploads/2023/12/what-is-concrete-body.jpg",
             use_container_width=True, caption="A Smarter Way to Engineer Concrete")

    st.markdown("""
    <h3 style='color:#4A90E2;'>📚 Dataset Source</h3>
    <p>
        This app uses the <strong>Concrete Compressive Strength Dataset</strong> from the 
        <a href="https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength" target="_blank">
        UCI Machine Learning Repository</a>. The dataset includes 1,030 observations with 8 input features and 1 output (strength).
    </p>
    """, unsafe_allow_html=True)

elif tab == "👨‍💻 About Me":
    st.write("# 🏅 Syed Muhammad Abdullah Abdulbadeeii")
    col1, col2, col3 = st.columns([4.5,1,4.5])
    with col1:
    # Personal Title 🏅🌟💡🌱🌍👤
        st.write("\n\n")
        st.markdown(
        "<img src='https://raw.githubusercontent.com/smaasui/SMAASU/main/16.jpeg' width='550'>",
        unsafe_allow_html=True)

        # st.image("https://raw.githubusercontent.com/smaasui/SMAASU/main/16.jpeg", use_container_width=True, width=100)
        # Expertise & Interests
        st.write("\n\n")
        st.write("# 🚀 Areas of Expertise")
        st.markdown(
            """
            - 🏗️ **Civil Engineering & Smart Infrastructure** – Engineering sustainable and innovative urban solutions.
            - 💻 **Software & Web Development** – Creating intelligent digital solutions to optimize efficiency.
            - 🤖 **Artificial Intelligence & Data Science** – Harnessing AI-driven technologies for smarter decision-making.
            - 📊 **Data Processing & Automation** – Streamlining complex workflows through advanced automation.
            - 🚀 **Entrepreneurship & Technological Innovation** – Spearheading startups that drive meaningful change.
            - ❤️ **Philanthropy & Social Impact** – Advocating for and supporting communities in need.
            """
        )


    with col3:
        st.write("# 🌱 About Me")
        # Introduction
        st.markdown(
            """
            I am **Syed Muhammad Abdullah Abdulbadeeii**, a **Civil Engineering Student at NED University of Engineering & Technology, Entrepreneur, Innovator, and Philanthropist**. 
            With a deep passion for **Artificial Intelligence, Architecture, and Sustainable Urbanization**, I am committed to pioneering **Transformative Solutions** that seamlessly integrate technology with real-world applications.
            
            My work is driven by a vision to **Build a Smarter, More Sustainable Future**, where cutting-edge innovations enhance efficiency, improve urban living, and empower businesses. 
            Beyond my professional pursuits, I am dedicated to **philanthropy**, striving to **uplift Muslims and support underprivileged communities**, fostering a society rooted in compassion, empowerment, and progress.
            """
        )
        
        # Vision & Journey
        st.write("# 🌍 My Vision & Journey")
        st.markdown(
            """
            As the founder of **SMAASU Corporation**, I have led groundbreaking initiatives such as **Data Duster**, a web-based platform revolutionizing data processing and automation. 
            My entrepreneurial journey is fueled by a relentless drive to **bridge the gap between technology and urban development**, delivering impactful solutions that **redefine the future of cities and industries**.
            
            **I believe in innovation, sustainability, and the power of technology to transform lives.** Through my work, I strive to create solutions that not only drive efficiency but also foster inclusivity and social well-being.
            
            **Let’s collaborate to build a brighter, more progressive future!**
            """
        )
        
    st.write("# 🔗 Engineering connections !")
    st.link_button("🔗 Stay connected on LinkedIn!", "https://www.linkedin.com/in/smaasui/")


elif tab == "🎓 About Us":

    # Company Title
    st.write("# 🏢 About SMAASU Corporation")

    # Introduction
    st.markdown(
        """
        **SMAASU Corporation** is a forward-thinking company committed to innovation in **technology, architecture, and sustainable urbanization**.
        Our vision is to create cutting-edge solutions that simplify workflows, enhance productivity, and contribute to a smarter, more efficient future.
        """
    )

    # Mission Section
    st.header("🌍 Our Mission")
    st.markdown(
        """
        At **SMAASU Corporation**, we aim to:
        - 🚀 **Develop pioneering software solutions** that enhance business efficiency.
        - 🏗️ **Revolutionize architecture and urban planning** with smart, sustainable designs.
        - 🌱 **Promote sustainability** in every project we undertake.
        - 🤝 **Empower businesses and individuals** with next-gen technology.
        """
    )

    # Core Values Section
    st.header("💡 Our Core Values")
    st.markdown(
        """
        - **Innovation** – Continuously pushing boundaries with cutting-edge technology.
        - **Sustainability** – Building a future that is eco-friendly and efficient.
        - **Excellence** – Delivering top-tier solutions with precision and quality.
        - **Integrity** – Upholding transparency and trust in every endeavor.
        """
    )

    # Call to Action
    st.markdown(
        """
        🚀 **Join us on our journey to create a smarter, more sustainable world with SMAASU Corporation!**
        """,
        unsafe_allow_html=True
    )

 
