import streamlit as st
import matplotlib
matplotlib.use('Agg')
st.set_page_config(
    page_title="ML Classification Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
import joblib
import os
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')
# ============================================
# CSS import
# ============================================
def load_css():
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# ============================================
# AUTHENTICATION - PERFECT CENTERED LOGIN
# ============================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.markdown("""
<style>
#MainMenu {visibility: hidden !important;}
header {visibility: hidden !important;}
footer {visibility: hidden !important;}
section[data-testid="stSidebar"] {display: none !important;}

.stApp {
    background: linear-gradient(135deg, #191970 0%, #1F305E 30%, #1a1a60 60%, #00009C 100%) !important;
    overflow-x: hidden !important;
}

.stApp > div:first-child { padding: 0 !important; margin: 0 !important; }
.main > div { padding: 0 !important; }
.block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }

[data-testid="stAppViewContainer"] {
    display: flex !important; align-items: center !important; justify-content: center !important;
    min-height: 100vh !important;
}

[data-testid="stAppViewContainer"] > .main {
    display: flex !important; align-items: center !important; justify-content: center !important;
    width: 100% !important;
}

[data-testid="stAppViewContainer"] > .main .block-container {
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
    min-height: 100vh !important; width: 100% !important;
}

div[class*="st-key-login_card"] {
    background: rgba(255, 255, 255, 0.98) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 20px !important; padding: 40px 35px !important;
    width: 100% !important; max-width: 420px !important;
    box-shadow: 0 25px 60px rgba(25, 25, 112, 0.5) !important;
    margin: 0 auto !important;
    border: 1px solid rgba(25, 25, 112, 0.1) !important;
}

.logo-animation { font-size: 3rem; text-align: center; animation: pulse 2s infinite; }

.stTextInput > div > div > input {
    border-radius: 12px !important; border: 2px solid #d0d0e0 !important;
    padding: 12px 16px !important; font-size: 1rem !important; background: #f8f9fa !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: #191970 !important; 
    box-shadow: 0 0 0 4px rgba(25, 25, 112, 0.15) !important;
    background: white !important;
}

.stButton > button, .stFormSubmitButton > button {
    border-radius: 12px !important; font-weight: 600 !important;
    padding: 12px 24px !important; width: 100% !important;
    transition: all 0.3s ease !important;
}

.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, #191970 0%, #2525a0 100%) !important;
    color: white !important; border: none !important;
    box-shadow: 0 8px 25px rgba(25, 25, 112, 0.4) !important;
}

.stFormSubmitButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(25, 25, 112, 0.6) !important;
}

.stFormSubmitButton > button[kind="secondary"] {
    background: white !important; color: #191970 !important;
    border: 2px solid #191970 !important;
}

.stFormSubmitButton > button[kind="secondary"]:hover {
    background: #f0f0ff !important;
}

.gradient-text {
    background: linear-gradient(135deg, #191970 0%, #483D8B 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-weight: 700;
}

@keyframes slideUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse { 0%,100%{transform:scale(1);} 50%{transform:scale(1.1);} }

@media (max-width: 480px) {
    div[class*="st-key-login_card"] { padding: 30px 20px !important; margin: 0 15px !important; }
}
</style>
""", unsafe_allow_html=True)

    # Login card
    with st.container(key="login_card"):
        st.markdown("""
        <div class="logo-animation">🤖</div>
        <h1 style="text-align: center; color: #1a1a2e; margin: 5px 0 2px; font-size: 1.8rem;">
            ML <span class="gradient-text">Pipeline</span>
        </h1>
        <p style="text-align: center; color: #888; margin-bottom: 25px; font-size: 0.85rem;">
            Machine Learning Platform
        </p>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            st.markdown('<div style="margin-top: 25px;"></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                login_btn = st.form_submit_button("🔑 Sign In", width="stretch", type="primary")
            with col2:
                guest_btn = st.form_submit_button("👤 Guest", width="stretch")

            if login_btn:
                if username == "admin" and password == "admin123":
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials!")
                    st.markdown('<p style="text-align:center;color:#999;font-size:0.8rem;">💡 <b>admin</b> / <b>admin123</b></p>', unsafe_allow_html=True)

            if guest_btn:
                st.session_state.logged_in = True
                st.session_state.username = "Guest"
                st.rerun()

    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <p style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">
            © 2026 Amir Khan | <a href="https://github.com/amirkhan421" style="color: #667eea; text-decoration: none;">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ============================================
# SIDEBAR - CONTROLS (CLEAN - NO DUPLICATES)
# ============================================
with st.sidebar:
    st.markdown("""
    <style>
    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F5F5FF 0%, #EBEBFF 50%, #E0E0FF 100%) !important;
        border-right: 2px solid rgba(25, 25, 112, 0.15) !important;
    }
    
    /* Text colors */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {
        color: #191970 !important;
        font-weight: 500 !important;
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #191970 !important;
        font-weight: 700 !important;
    }
    
    /* Selectbox */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background: white !important;
        border: 2px solid rgba(25, 25, 112, 0.3) !important;
        border-radius: 10px !important;
        color: #191970 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:focus {
        border-color: #191970 !important;
        box-shadow: 0 0 0 3px rgba(25, 25, 112, 0.15) !important;
    }
    
/* ==============================
   SIDEBAR SLIDER
============================== */

/* Slider Track */
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div {
    background: #c723d9 !important;
}


/* Active Slider Track */
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
    background: #191970 !important;
}


/* Slider Thumb */
section[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: #191970 !important;
    border: 3px solid #FFFFFF !important;
    box-shadow: 0 0 8px rgba(25,25,112,0.45) !important;
}
    
    /* Number Input */
    section[data-testid="stSidebar"] .stNumberInput input {
        background: white !important;
        border: 2px solid rgba(25, 25, 112, 0.3) !important;
        color: #191970 !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    section[data-testid="stSidebar"] .stNumberInput input:focus {
        border-color: #191970 !important;
        box-shadow: 0 0 0 3px rgba(25, 25, 112, 0.15) !important;
    }
    
    section[data-testid="stSidebar"] .stNumberInput button {
        color: #191970 !important;
        background: rgba(25, 25, 112, 0.08) !important;
        border: 1px solid rgba(25, 25, 112, 0.2) !important;
    }
    
    /* Buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #191970 0%, #2525a0 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(25, 25, 112, 0.3) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #2525a0 0%, #3a3ab0 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(25, 25, 112, 0.5) !important;
    }
    
    /* Checkbox */
    section[data-testid="stSidebar"] .stCheckbox label span {
        color: #191970 !important;
    }
    
    /* Divider */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(25, 25, 112, 0.15) !important;
    }
    
    /* User Section */
    .sidebar-user-section {
        background: linear-gradient(135deg, rgba(25, 25, 112, 0.06), rgba(72, 61, 139, 0.1)) !important;
        border: 2px solid rgba(25, 25, 112, 0.25) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        text-align: center;
        margin-top: 20px;
    }
    
    .sidebar-user-section:hover {
        background: linear-gradient(135deg, rgba(25, 25, 112, 0.1), rgba(72, 61, 139, 0.15)) !important;
        border-color: #191970 !important;
        box-shadow: 0 6px 20px rgba(25, 25, 112, 0.25) !important;
    }
    
    .sidebar-user-section .user-avatar {
        font-size: 2rem;
        margin-bottom: 5px;
    }
    
    .sidebar-user-section .user-name {
        color: #191970 !important;
        font-weight: 700 !important;
        font-size: 0.95rem;
    }
    
    .sidebar-user-section .user-role {
        color: #483D8B !important;
        font-size: 0.78rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ============================================
    # SIDEBAR HEADER
    # ============================================
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("assets/machine-learning.png", width=45)
    with col2:
        st.markdown("""
        <div style="padding-top:8px;">
            <div style="font-size:1.1rem;font-weight:700;color:#191970;">ML Pipeline</div>
            <div style="font-size:0.7rem;color:#191970;">Classification Dashboard</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎮 Controls")
    
    st.markdown("#### 📊 Dataset")
    dataset_option = st.selectbox("Select Dataset", ["Iris (Built-in)", "Upload Custom CSV"], label_visibility="collapsed")
    
    st.markdown("---")
    
    st.markdown("#### 🔧 Parameters")
    test_size = st.slider("Test Size (%)", 10, 50, 20, 5)
    random_state = st.number_input("Random State", 0, 100, 42)
    
    st.markdown("---")
    
    st.markdown("#### 🤖 Models")
    train_rf = st.checkbox("Random Forest", value=True)
    train_svm = st.checkbox("SVM", value=True)
    
    st.markdown("---")
    
    st.markdown("#### ⚙️ Tuning")
    enable_tuning = st.checkbox("Enable GridSearchCV", value=True)
    cv_folds = st.slider("CV Folds", 3, 10, 5)
    
    st.markdown("---")
    
    train_button = st.button("🚀 Train Models", type="primary", width="stretch")
    
    st.markdown("---")
    
    st.markdown("### 👨‍💻 Author")
    st.markdown("""
    <div style="color:#1B1B1B;font-size:0.85rem;">
    <b>Amir Khan</b><br>
    <a href="https://github.com/amirkhan421" style="color:#1B1B1B;text-decoration:none;">GitHub ↗</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # User section at bottom
    st.markdown(f"""
    <div class="sidebar-user-section">
        <div class="user-avatar">👤</div>
        <div class="user-name">{st.session_state.get('username', 'User')}</div>
        <div class="user-role">Logged in</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Logout", width="stretch"):
        st.session_state.logged_in = False
        st.rerun()



        
# ============================================
# MAIN CONTENT
# ============================================
st.markdown('<p class="main-header">🤖 Machine Learning Classification Pipeline</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Complete Supervised Learning Workflow with Random Forest & SVM</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Analysis", "🔧 Preprocessing", "🤖 Model Training", "📈 Results", "🎯 Predictions"])

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# ============================================
# TAB 1: DATA ANALYSIS
# ============================================
with tab1:
    st.header("📊 Data Exploration & Analysis")
    
    if dataset_option == "Iris (Built-in)":
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        st.session_state.df = df
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            st.session_state.df = pd.read_csv(uploaded_file)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Metrics row - 4 columns (ye theek hai chhota hone ki wajah se)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Samples", df.shape[0])
        with col2:
            st.metric("🔢 Features", df.shape[1] - 2)
        with col3:
            st.metric("🎯 Classes", df['species'].nunique())
        with col4:
            st.metric("❌ Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # 📋 Dataset Preview - FULL WIDTH
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), width="stretch")
        
        st.markdown("---")
        
        # 📊 Statistical Summary - FULL WIDTH
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe(), width="stretch")
        
        st.markdown("---")
        
        # 📈 Class Distribution - FULL WIDTH
        st.subheader("📈 Class Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        df['species'].value_counts().plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_xlabel('Species')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution', fontweight='bold')
        st.pyplot(fig)
        
        st.markdown("---")
        
        # 🔍 Class Balance - FULL WIDTH
        st.subheader("🔍 Class Balance")
        balance_df = pd.DataFrame({
            'Species': df['species'].value_counts().index,
            'Count': df['species'].value_counts().values,
            'Percentage': (df['species'].value_counts(normalize=True) * 100).values
        })
        st.dataframe(balance_df, width="stretch")

# ============================================
# TAB 2: PREPROCESSING
# ============================================
with tab2:
    st.header("🔧 Data Preprocessing")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.subheader("Original Data Info")
        st.code(f"Shape: {df.shape}\nMissing Values: {df.isnull().sum().sum()}\nFeatures: {list(df.columns)}")
        
        st.subheader("🔄 Preprocessing Pipeline")
        steps = """
        Pipeline Steps:
        1. ✨ Feature Selection: Separating features from target
        2. 🎯 Target Encoding: Label encoding for classification
        3. 📊 Train-Test Split: Stratified 80-20 split
        4. 🔧 Missing Value Imputation: Median strategy
        5. 📏 Feature Scaling: StandardScaler for normalization
        """
        st.code(steps)
        
        if st.button("🔍 Preview Preprocessing"):
            X = df.iloc[:, :-2]
            y = df['target']
            
            np.random.seed(42)
            mask = np.random.random(X.shape) < 0.05
            X_masked = X.copy()
            X_masked[mask] = np.nan
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before Preprocessing (with NaN)**")
                st.dataframe(X_masked.head(), width="stretch")
                st.write(f"Missing values: {X_masked.isnull().sum().sum()}")
            
            with col2:
                imputer = SimpleImputer(strategy='median')
                scaler = StandardScaler()
                X_imputed = imputer.fit_transform(X_masked)
                X_scaled = scaler.fit_transform(X_imputed)
                X_processed = pd.DataFrame(X_scaled, columns=X.columns)
                
                st.write("**After Preprocessing (Clean & Scaled)**")
                st.dataframe(X_processed.head(), width="stretch")
                st.write(f"Missing values: 0 ✅")

# ============================================
# TAB 3: MODEL TRAINING
# ============================================
with tab3:
    st.header("🤖 Model Training")
    
    if st.session_state.df is not None and train_button:
        df = st.session_state.df
        X = df.iloc[:, :-2]
        y = df['target']
        
        np.random.seed(random_state)
        mask = np.random.random(X.shape) < 0.05
        X_masked = X.copy()
        X_masked[mask] = np.nan
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_masked, y, test_size=test_size/100, random_state=random_state, stratify=y
        )
        
        preprocessor = ColumnTransformer(
            transformers=[('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), X.columns.tolist())]
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        if train_rf:
            status_text.text("🔄 Training Random Forest...")
            progress_bar.progress(20)
            
            rf_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=random_state))
            ])
            rf_pipeline.fit(X_train, y_train)
            rf_pred = rf_pipeline.predict(X_test)
            rf_acc = accuracy_score(y_test, rf_pred)
            
            results['Random Forest (Baseline)'] = {
                'pipeline': rf_pipeline,
                'accuracy': rf_acc,
                'predictions': rf_pred
            }
            
            progress_bar.progress(40)
            
            if enable_tuning:
                status_text.text("⚙️ Tuning Random Forest...")
                rf_param_grid = {
                    'classifier__n_estimators': [50, 100],
                    'classifier__max_depth': [None, 10],
                    'classifier__min_samples_split': [2, 5]
                }
                rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                rf_grid.fit(X_train, y_train)
                rf_tuned_pred = rf_grid.predict(X_test)
                rf_tuned_acc = accuracy_score(y_test, rf_tuned_pred)
                
                results['Random Forest (Tuned)'] = {
                    'pipeline': rf_grid.best_estimator_,
                    'accuracy': rf_tuned_acc,
                    'predictions': rf_tuned_pred,
                    'best_params': rf_grid.best_params_
                }
            
            progress_bar.progress(60)
        
        if train_svm:
            status_text.text("🔄 Training SVM...")
            
            svm_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', SVC(random_state=random_state, probability=True))
            ])
            svm_pipeline.fit(X_train, y_train)
            svm_pred = svm_pipeline.predict(X_test)
            svm_acc = accuracy_score(y_test, svm_pred)
            
            results['SVM (Baseline)'] = {
                'pipeline': svm_pipeline,
                'accuracy': svm_acc,
                'predictions': svm_pred
            }
            
            progress_bar.progress(80)
            
            if enable_tuning:
                status_text.text("⚙️ Tuning SVM...")
                svm_param_grid = {
                    'classifier__C': [1, 10],
                    'classifier__gamma': ['scale', 'auto'],
                    'classifier__kernel': ['rbf']
                }
                svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                svm_grid.fit(X_train, y_train)
                svm_tuned_pred = svm_grid.predict(X_test)
                svm_tuned_acc = accuracy_score(y_test, svm_tuned_pred)
                
                results['SVM (Tuned)'] = {
                    'pipeline': svm_grid.best_estimator_,
                    'accuracy': svm_tuned_acc,
                    'predictions': svm_tuned_pred,
                    'best_params': svm_grid.best_params_
                }
            
            progress_bar.progress(100)
        
        st.session_state.results = results
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.trained = True
        
        status_text.text("✅ Training Complete!")
        progress_bar.empty()
        
        st.success("🎉 Models trained successfully!")
        
        st.subheader("📊 Training Results")
        results_df = pd.DataFrame([
            {'Model': name, 'Accuracy': f"{info['accuracy']:.4f} ({info['accuracy']*100:.2f}%)"}
            for name, info in results.items()
        ])
        st.dataframe(results_df, width="stretch", hide_index=True)
    else:
        st.info("👈 Click 'Train Models' in sidebar to start training")

# ============================================
# TAB 4: RESULTS - DETAILED
# ============================================
with tab4:
    st.header("📈 Results & Evaluation")
    
    if st.session_state.trained:
        results = st.session_state.results
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model = results[best_model_name]
        improvement = (best_model['accuracy'] - list(results.values())[0]['accuracy']) * 100
        
        # ============================================
        # TOP METRICS CARDS
        # ============================================
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">'
                       f'<h3>🏆 Best Model</h3>'
                       f'<h2>{best_model_name}</h2>'
                       '</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="success-card">'
                       f'<h3>🎯 Accuracy</h3>'
                       f'<h2>{best_model["accuracy"]*100:.2f}%</h2>'
                       '</div>', unsafe_allow_html=True)
        
        with col3:
            delta_color = "normal" if improvement >= 0 else "inverse"
            st.markdown('<div class="metric-card">'
                       f'<h3>📈 Improvement</h3>'
                       f'<h2>{improvement:.2f}%</h2>'
                       '</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="success-card">'
                       f'<h3>📊 Total Models</h3>'
                       f'<h2>{len(results)}</h2>'
                       '</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ============================================
        # SECTION 1: ALL MODELS PERFORMANCE TABLE
        # ============================================
        st.subheader("📊 All Models Performance")
        
        # Detailed results table
        results_detail = []
        for name, info in results.items():
            model_type = "Tuned" if "Tuned" in name else "Baseline"
            model_base = name.replace(" (Baseline)", "").replace(" (Tuned)", "")
            results_detail.append({
                'Model': model_base,
                'Type': model_type,
                'Accuracy': f"{info['accuracy']*100:.2f}%",
                'Accuracy_Val': info['accuracy'],
                'Status': '🏆 Best' if name == best_model_name else ''
            })
        
        results_df = pd.DataFrame(results_detail)
        
        # Color code the best model
        def highlight_best(row):
            if row['Status'] == '🏆 Best':
                return ['background-color: #d4edda; color: #155724'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            results_df[['Model', 'Type', 'Accuracy', 'Status']].style.apply(highlight_best, axis=1),
            width="stretch", hide_index=True
        )
        
        st.markdown("---")
        
        # ============================================
        # SECTION 2: MODEL COMPARISON CHART
        # ============================================
        st.subheader("📊 Model Comparison Chart")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            names = list(results.keys())
            accuracies = [results[n]['accuracy'] for n in names]
            colors = ['#FF6B6B' if 'Baseline' in n else '#4ECDC4' for n in names]
            
            bars = ax.bar(names, accuracies, color=colors, edgecolor='white', linewidth=0.5)
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.set_ylim([0.75, 1.05])
            ax.axhline(y=0.92, color='red', linestyle='--', linewidth=1.5, label='92% Target')
            ax.axhline(y=best_model['accuracy'], color='green', linestyle=':', linewidth=1, label=f'Best: {best_model["accuracy"]*100:.2f}%')
            ax.legend(loc='lower right')
            ax.grid(axis='y', alpha=0.3)
            
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                       f'{acc:.3f}', ha='center', fontweight='bold', fontsize=10)
            
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### 📋 Legend
            - 🔴 **Red Bars** = Baseline Models
            - 🟢 **Green Bars** = Tuned Models
            - 🔴 **Red Line** = 92% Target
            - 🟢 **Green Dotted** = Best Model Score
            
            ### 💡 Insight
            Hyperparameter tuning improves model performance by finding optimal parameters.
            """)
            
            # Show best parameters if available
            if 'best_params' in best_model:
                st.markdown("### ⚙️ Best Parameters")
                st.json(best_model['best_params'])
        
        st.markdown("---")
        
        # ============================================
        # SECTION 3: CONFUSION MATRIX + METRICS
        # ============================================
        st.subheader("🎯 Confusion Matrix & Per-Class Metrics")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Confusion Matrix
            fig, ax = plt.subplots(figsize=(7, 6))
            iris = load_iris()
            ConfusionMatrixDisplay.from_estimator(
                best_model['pipeline'], X_test, y_test,
                display_labels=iris.target_names, ax=ax, cmap='Blues',
                colorbar=False
            )
            ax.set_title(f'{best_model_name} - Confusion Matrix', fontweight='bold', fontsize=14)
            st.pyplot(fig)
        
        with col2:
            # Per-Class Metrics
            from sklearn.metrics import classification_report
            import io
            
            y_pred = best_model['pipeline'].predict(X_test)
            report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
            
            st.markdown("### 📋 Classification Report")
            
            # Create metrics table
            metrics_data = []
            for class_name in iris.target_names:
                m = report[class_name]
                metrics_data.append({
                    'Class': class_name,
                    'Precision': f"{m['precision']:.3f}",
                    'Recall': f"{m['recall']:.3f}",
                    'F1-Score': f"{m['f1-score']:.3f}",
                    'Support': int(m['support'])
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, width="stretch", hide_index=True)
            
            # Accuracy summary
            st.markdown("### 📊 Overall Metrics")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Accuracy", f"{report['accuracy']*100:.2f}%")
                st.metric("Macro Avg F1", f"{report['macro avg']['f1-score']:.3f}")
            with col_b:
                st.metric("Weighted Avg F1", f"{report['weighted avg']['f1-score']:.3f}")
                st.metric("Samples", int(report['macro avg']['support']))
        
        st.markdown("---")
        
        # ============================================
        # SECTION 4: CROSS-VALIDATION
        # ============================================
        st.subheader("✅ Cross-Validation Analysis")
        
        cv_scores = cross_val_score(best_model['pipeline'], X_test, y_test, cv=5)
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            # CV Scores visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
            
            bars = ax.bar(folds, cv_scores, color=['#667eea', '#764ba2', '#11998e', '#38ef7d', '#2193b0'])
            ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.set_ylim([0.7, 1.05])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            for bar, score in zip(bars, cv_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{score:.3f}', ha='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # CV Stats
            st.markdown("### 📊 CV Statistics")
            cv_df = pd.DataFrame({
                'Fold': folds,
                'Accuracy': [f'{s:.4f}' for s in cv_scores],
                'Deviation': [f'{s - cv_scores.mean():+.4f}' for s in cv_scores]
            })
            st.dataframe(cv_df, width="stretch", hide_index=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
                st.metric("Min Score", f"{cv_scores.min():.4f}")
            with col_b:
                st.metric("Std Deviation", f"{cv_scores.std():.4f}")
                st.metric("Max Score", f"{cv_scores.max():.4f}")
            
            # Stability indicator
            std = cv_scores.std()
            if std < 0.05:
                st.success("🟢 Very Stable Model")
            elif std < 0.1:
                st.warning("🟡 Moderately Stable")
            else:
                st.error("🔴 High Variance")
        
        st.markdown("---")
        
        # ============================================
        # SECTION 5: DOWNLOAD & EXPORT
        # ============================================
        st.subheader("💾 Save & Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Best Model", width="stretch", type="primary"):
                os.makedirs('models', exist_ok=True)
                joblib.dump(best_model['pipeline'], 'models/best_model.pkl')
                st.success("✅ Model saved to `models/best_model.pkl`")
        
        with col2:
            # Export all results as CSV
            results_export = pd.DataFrame([
                {'Model': name, 'Accuracy': info['accuracy'], 'Type': 'Tuned' if 'Tuned' in name else 'Baseline'}
                for name, info in results.items()
            ])
            csv_data = results_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_data,
                file_name="ml_results.csv",
                mime="text/csv",
                width="stretch"
            )
        
        with col3:
            # Export classification report
            report_text = classification_report(y_test, y_pred, target_names=iris.target_names)
            st.download_button(
                label="📥 Download Full Report",
                data=report_text,
                file_name="classification_report.txt",
                mime="text/plain",
                width="stretch"
            )
        
        st.markdown("---")
        
        # ============================================
        # SECTION 6: SUMMARY INSIGHTS
        # ============================================
        st.subheader("💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            ### 🏆 Best Model: {best_model_name}
            - **Accuracy:** {best_model['accuracy']*100:.2f}%
            - **CV Mean:** {cv_scores.mean():.4f}
            - **CV Std:** {cv_scores.std():.4f}
            - **Improvement:** {improvement:.2f}% over baseline
            """)
        
        with col2:
            # Compare tuned vs baseline
            base_models = {k: v for k, v in results.items() if 'Baseline' in k}
            tuned_models = {k: v for k, v in results.items() if 'Tuned' in k}
            
            if base_models and tuned_models:
                avg_base = np.mean([v['accuracy'] for v in base_models.values()])
                avg_tuned = np.mean([v['accuracy'] for v in tuned_models.values()])
                
                st.markdown(f"""
                ### 📈 Tuning Impact
                - **Avg Baseline Accuracy:** {avg_base*100:.2f}%
                - **Avg Tuned Accuracy:** {avg_tuned*100:.2f}%
                - **Tuning Boost:** {(avg_tuned - avg_base)*100:.2f}%
                - **Models Compared:** {len(results)}
                """)
        
    else:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 4rem;">📈</div>
            <h2>No Results Yet</h2>
            <p style="color: #666;">Train your models first from the <b>Model Training</b> tab!</p>
        </div>
        """, unsafe_allow_html=True)

        # ============================================
# TAB 5: PREDICTIONS
# ============================================
with tab5:
    st.header("🎯 Make Predictions")
    
    iris = load_iris()
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Model source selection
    model_source = st.radio(
        "🤖 Select Model Source",
        ["📦 Use Trained Model", "💾 Load Saved Model", "🔄 Quick Train (Demo)"],
        horizontal=True
    )
    
    model = None
    
    if model_source == "📦 Use Trained Model":
        if 'trained' in st.session_state and st.session_state.trained:
            results = st.session_state.results
            best_model_name = max(results, key=lambda x: results[x]['accuracy'])
            if 'pipeline' in results[best_model_name]:
                model = results[best_model_name]['pipeline']
                st.success(f"✅ Using: {best_model_name} ({results[best_model_name]['accuracy']*100:.2f}%)")
        else:
            st.warning("⚠️ No trained model! Train models first or use another source.")
    
    elif model_source == "💾 Load Saved Model":
        col1, col2 = st.columns([3, 1])
        with col1:
            model_path = st.text_input("Model Path", value="models/best_model.pkl", key="pred_model_path")
        with col2:
            st.write("")
            st.write("")
            load_btn = st.button("📂 Load", width="stretch")
        
        if load_btn or (model_path and os.path.exists(model_path)):
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    st.success(f"✅ Model loaded from: {os.path.basename(model_path)}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning(f"⚠️ File not found: {model_path}")
    
    elif model_source == "🔄 Quick Train (Demo)":
        if st.button("🚀 Quick Train Model", type="primary"):
            with st.spinner("Training quick model..."):
                iris_data = load_iris()
                X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
                y = iris_data.target
                preprocessor = ColumnTransformer(
                    transformers=[('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), iris_data.feature_names)]
                )
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
                model.fit(X, y)
                st.success("✅ Quick model trained! Accuracy: ~97%")
    
    st.markdown("---")
    
    # Prediction method
    pred_method = st.radio(
        "🎯 Choose Prediction Method",
        ["📝 Manual Input", "📤 Batch Upload CSV", "🎲 Random Sample"],
        horizontal=True
    )
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Manual Input
    if pred_method == "📝 Manual Input":
        st.subheader("📝 Enter Feature Values")
        
        col1, col2 = st.columns(2)
        with col1:
            sl = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
            sw = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        with col2:
            pl = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.3, 0.1)
            pw = st.number_input("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)
        
        if st.button("🔮 Predict", type="primary", width="stretch"):
            if model is not None:
                try:
                    input_data = pd.DataFrame([[sl, sw, pl, pw]], columns=feature_names)
                    pred = model.predict(input_data)[0]
                    try:
                        proba = model.predict_proba(input_data)[0]
                    except:
                        proba = None
                    
                    st.session_state.prediction_history.append({
                        'Time': datetime.now().strftime('%H:%M:%S'),
                        'Prediction': target_names[pred],
                        'Confidence': f"{max(proba)*100:.1f}%" if proba is not None else "N/A"
                    })
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
                        color = colors[pred] if pred < len(colors) else '#667eea'
                        st.markdown(f"""
                        <div style="background:{color};padding:30px;border-radius:15px;text-align:center;">
                            <h2 style="color:white;">{target_names[pred].upper()}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        if proba is not None:
                            mp = max(proba)
                            if mp > 0.8:
                                st.success(f"🟢 High Confidence: {mp:.1%}")
                            elif mp > 0.5:
                                st.warning(f"🟡 Medium Confidence: {mp:.1%}")
                            else:
                                st.error(f"🔴 Low Confidence: {mp:.1%}")
                    
                    with col2:
                        if proba is not None:
                            fig, ax = plt.subplots()
                            ax.bar(target_names, proba, color=['#4ECDC4','#FF6B6B','#45B7D1'])
                            ax.set_ylim([0,1])
                            for i, p in enumerate(proba):
                                ax.text(i, p+0.02, f'{p:.2%}', ha='center', fontweight='bold')
                            st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("⚠️ No model available!")
    
    # Batch Upload
    elif pred_method == "📤 Batch Upload CSV":
        st.subheader("📤 Upload CSV for Batch Predictions")
        st.code("sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n5.1,3.5,1.4,0.2\n7.0,3.2,4.7,1.4")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key="batch_upload")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.dataframe(batch_df.head(), width="stretch")
            
            if st.button("🔮 Predict Batch", type="primary"):
                if model is not None:
                    predictions = model.predict(batch_df)
                    batch_df['Predicted'] = [target_names[p] for p in predictions]
                    try:
                        proba = model.predict_proba(batch_df)
                        batch_df['Confidence'] = [f"{max(p)*100:.1f}%" for p in proba]
                    except:
                        batch_df['Confidence'] = 'N/A'
                    
                    st.subheader("🎯 Results")
                    st.dataframe(batch_df, width="stretch")
                    
                    csv_data = batch_df.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv_data, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
                else:
                    st.warning("⚠️ No model available!")
    
    # Random Sample
    elif pred_method == "🎲 Random Sample":
        st.subheader("🎲 Generate Random Samples")
        num_samples = st.slider("Number of Samples", 1, 30, 5)
        
        if st.button("🎲 Generate & Predict", type="primary"):
            np.random.seed(int(datetime.now().timestamp()))
            random_data = pd.DataFrame({
                'sepal length (cm)': np.random.uniform(4.3, 7.9, num_samples).round(1),
                'sepal width (cm)': np.random.uniform(2.0, 4.4, num_samples).round(1),
                'petal length (cm)': np.random.uniform(1.0, 6.9, num_samples).round(1),
                'petal width (cm)': np.random.uniform(0.1, 2.5, num_samples).round(1)
            })
            
            if model is not None:
                predictions = model.predict(random_data)
                random_data['Predicted'] = [target_names[p] for p in predictions]
                st.subheader("🎯 Results")
                st.dataframe(random_data, width="stretch")
                
                fig, ax = plt.subplots()
                random_data['Predicted'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#4ECDC4','#FF6B6B','#45B7D1'])
                ax.set_ylabel('')
                st.pyplot(fig)
            else:
                st.warning("⚠️ No model available!")
    
    # Prediction History
    st.markdown("---")
    st.subheader("📜 Prediction History")
    if st.session_state.prediction_history:
        st.dataframe(pd.DataFrame(st.session_state.prediction_history), width="stretch")
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions yet.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
    <p>🤖 Machine Learning Classification Pipeline | Built with Streamlit & Scikit-learn</p>
    <p>© 2026 Amir Khan | <a href="https://github.com/amirkhan421">GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True
)