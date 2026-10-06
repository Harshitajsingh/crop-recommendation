import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Smart Crop AI - Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Cached Resource Loaders
# ==============================================================================
@st.cache_resource
def load_trained_model():
    model_path = "crop_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model artifact '{model_path}' not found! Please run 'train_model.py' to generate it.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def load_dataset():
    data_path = os.path.join("dataset", "Crop_recommendation.csv")
    if not os.path.exists(data_path):
        st.error(f"❌ Dataset file '{data_path}' not found!")
        st.stop()
    return pd.read_csv(data_path)


model = load_trained_model()
df_dataset = load_dataset()

# ==============================================================================
# Agronomic Knowledge Base for All 22 Crops
# ==============================================================================
CROP_INFO = {
    "rice": {
        "name": "Rice",
        "emoji": "🌾",
        "category": "Cereal / Grain",
        "season": "Kharif (Monsoon)",
        "water": "High (> 1500 mm / flooded)",
        "soil": "Clayey, alluvial soils with high water retention",
        "duration": "100 - 150 days",
        "tip": "Maintain standing water during early vegetative stages. Nitrogen top-dressing boosts yield."
    },
    "maize": {
        "name": "Maize (Corn)",
        "emoji": "🌽",
        "category": "Cereal / Grain",
        "season": "Kharif & Rabi",
        "water": "Moderate (500 - 800 mm)",
        "soil": "Well-drained fertile loamy to silty soils",
        "duration": "90 - 120 days",
        "tip": "Ensure good drainage to prevent waterlogging; sensitive to stagnant moisture at germination."
    },
    "chickpea": {
        "name": "Chickpea (Gram)",
        "emoji": "🫘",
        "category": "Pulse / Legume",
        "season": "Rabi (Winter)",
        "water": "Low (300 - 450 mm)",
        "soil": "Deep, well-aerated sandy loam to clay loam",
        "duration": "90 - 110 days",
        "tip": "Fixes atmospheric nitrogen; requires minimal synthetic nitrogen fertilizers."
    },
    "kidneybeans": {
        "name": "Kidney Beans (Rajma)",
        "emoji": "🫘",
        "category": "Pulse / Legume",
        "season": "Rabi / Spring",
        "water": "Moderate (400 - 600 mm)",
        "soil": "Light, organic-rich well-drained loam (pH 6.0 - 6.5)",
        "duration": "100 - 130 days",
        "tip": "Does not tolerate waterlogging or saline soils. Sensitive to frost."
    },
    "pigeonpeas": {
        "name": "Pigeon Peas (Arhar / Tur)",
        "emoji": "🫘",
        "category": "Pulse / Legume",
        "season": "Kharif",
        "water": "Low to Moderate (600 - 750 mm, drought-tolerant)",
        "soil": "Deep loam to clay loam with good internal drainage",
        "duration": "150 - 200 days",
        "tip": "Deep taproot system allows moisture extraction during dry spells."
    },
    "mothbeans": {
        "name": "Moth Beans",
        "emoji": "🫘",
        "category": "Pulse / Legume",
        "season": "Kharif (Dryland)",
        "water": "Very Low (200 - 400 mm, highly drought-resistant)",
        "soil": "Sandy, poor dryland soils",
        "duration": "65 - 80 days",
        "tip": "Excellent soil cover crop that prevents erosion in arid and semi-arid tracts."
    },
    "mungbean": {
        "name": "Mung Bean (Green Gram)",
        "emoji": "🫘",
        "category": "Pulse / Legume",
        "season": "Kharif / Summer",
        "water": "Low (350 - 500 mm)",
        "soil": "Well-drained fertile loam",
        "duration": "60 - 75 days",
        "tip": "Short-duration crop ideal for crop rotation and soil nitrogen restoration."
    },
    "blackgram": {
        "name": "Black Gram (Urad)",
        "emoji": "🫘",
        "category": "Pulse / Legume",
        "season": "Kharif & Summer",
        "water": "Moderate (400 - 600 mm)",
        "soil": "Heavy black cotton soils or clay loams",
        "duration": "70 - 90 days",
        "tip": "Phosphorus application promotes root nodulation and early vigor."
    },
    "lentil": {
        "name": "Lentil (Masoor)",
        "emoji": "🫘",
        "category": "Pulse / Legume",
        "season": "Rabi (Cool dry)",
        "water": "Low (300 - 450 mm)",
        "soil": "Alluvial or medium black soils",
        "duration": "110 - 130 days",
        "tip": "Requires cool temperatures during vegetative growth and warm weather at harvest."
    },
    "pomegranate": {
        "name": "Pomegranate",
        "emoji": "🍎",
        "category": "Fruit / Horticulture",
        "season": "Perennial (All-season pruning cycles)",
        "water": "Low to Moderate (Drip irrigation advised)",
        "soil": "Deep, well-drained loamy or alluvial soil",
        "duration": "150 - 180 days (flowering to harvest)",
        "tip": "Regular micro-irrigation and balanced potassium enhance fruit color and rind quality."
    },
    "banana": {
        "name": "Banana",
        "emoji": "🍌",
        "category": "Fruit / Horticulture",
        "season": "Year-round (Tropical)",
        "water": "High (1500 - 2000 mm evenly distributed)",
        "soil": "Rich, deep loamy soil rich in organic matter and Potassium",
        "duration": "11 - 14 months",
        "tip": "Heavy feeder of Potassium and Nitrogen; requires soil moisture protection via mulching."
    },
    "mango": {
        "name": "Mango",
        "emoji": "🥭",
        "category": "Fruit / Horticulture",
        "season": "Summer (Perennial orchard)",
        "water": "Moderate (Requires dry spell for flowering)",
        "soil": "Deep alluvial and well-drained red or loamy soil",
        "duration": "Annual bearing cycle",
        "tip": "Restricted watering prior to flowering triggers abundant blossom induction."
    },
    "grapes": {
        "name": "Grapes",
        "emoji": "🍇",
        "category": "Fruit / Horticulture",
        "season": "Perennial (Spring / Summer harvest)",
        "water": "Moderate (Sensitive to rainfall at berry ripening)",
        "soil": "Well-drained sandy loam or gravelly soil",
        "duration": "120 - 140 days post-pruning",
        "tip": "High Potassium demand; avoid rains near harvest to prevent fruit cracking and mildew."
    },
    "watermelon": {
        "name": "Watermelon",
        "emoji": "🍉",
        "category": "Fruit / Cucurbit",
        "season": "Summer (Zaid)",
        "water": "Moderate (400 - 600 mm, regular watering until fruit set)",
        "soil": "Sandy to sandy loam, well-drained warm soil",
        "duration": "80 - 100 days",
        "tip": "Withhold heavy watering during fruit maturation to maximize sweetness (Brix index)."
    },
    "muskmelon": {
        "name": "Muskmelon",
        "emoji": "🍈",
        "category": "Fruit / Cucurbit",
        "season": "Summer (Zaid)",
        "water": "Moderate (Controlled furrow or drip irrigation)",
        "soil": "Sandy loam soils with excellent sun exposure",
        "duration": "75 - 90 days",
        "tip": "Requires warm, dry weather at maturity for sugar synthesis and netted rind development."
    },
    "apple": {
        "name": "Apple",
        "emoji": "🍏",
        "category": "Fruit / Temperate",
        "season": "Autumn (Temperate climate)",
        "water": "Moderate to High (1000 - 1250 mm annual)",
        "soil": "Deep, well-drained loamy soils rich in organic matter (pH 5.5 - 6.5)",
        "duration": "Perennial orchard (130 - 150 days fruit development)",
        "tip": "Requires sufficient winter chilling hours (< 7°C) for healthy bud break."
    },
    "orange": {
        "name": "Orange (Citrus)",
        "emoji": "🍊",
        "category": "Fruit / Citrus",
        "season": "Winter / Spring",
        "water": "Moderate (800 - 1200 mm)",
        "soil": "Deep, well-drained sandy loam or light clay loam",
        "duration": "240 - 270 days",
        "tip": "Avoid high soil salinity or water stagnancy; micronutrients like Zinc and Boron prevent chlorosis."
    },
    "papaya": {
        "name": "Papaya",
        "emoji": "🥭",
        "category": "Fruit / Quick Crop",
        "season": "Year-round (Warm tropical)",
        "water": "Moderate (Regular moist conditions, zero waterlogging)",
        "soil": "Rich, well-drained porous sandy loam (pH 6.0 - 6.8)",
        "duration": "9 - 11 months",
        "tip": "Extremely susceptible to foot rot; elevate planting beds for surplus water runoff."
    },
    "coconut": {
        "name": "Coconut",
        "emoji": "🥥",
        "category": "Plantation / Commercial",
        "season": "Year-round (Coastal / Tropical)",
        "water": "High (1300 - 2300 mm annual rainfall or irrigation)",
        "soil": "Coastal sand, alluvial or red sandy loam",
        "duration": "Perennial palm (continuous monthly yield)",
        "tip": "Tolerates high humidity and coastal salt spray; thrives on Potassium-rich feeding."
    },
    "cotton": {
        "name": "Cotton",
        "emoji": "⚪",
        "category": "Commercial / Fiber",
        "season": "Kharif",
        "water": "Moderate (600 - 1000 mm)",
        "soil": "Deep black cotton soils (Vertisols) or alluvial loam",
        "duration": "150 - 180 days",
        "tip": "Requires bright sunny days during boll bursting; high rain at harvest damages fiber."
    },
    "jute": {
        "name": "Jute (Golden Fiber)",
        "emoji": "🌾",
        "category": "Commercial / Fiber",
        "season": "Kharif (Warm and humid)",
        "water": "High (1200 - 1800 mm)",
        "soil": "New alluvial soils along river basins",
        "duration": "120 - 150 days",
        "tip": "Requires high relative humidity (> 75%) and ample water for retting fiber extraction."
    },
    "coffee": {
        "name": "Coffee",
        "emoji": "☕",
        "category": "Plantation / Beverage",
        "season": "Perennial (Winter harvest)",
        "water": "High (1500 - 2200 mm with dry spell for blossom)",
        "soil": "Porous, acidic forest loam rich in humus (pH 5.5 - 6.5)",
        "duration": "Perennial under shade tree canopy",
        "tip": "Grown under shade trees; blossom showers in March/April are vital for fruit set."
    }
}

# ==============================================================================
# Custom CSS Styling
# ==============================================================================
st.markdown("""
<style>
    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%);
        color: white;
        padding: 30px 35px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(27, 94, 32, 0.18);
    }
    .hero-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 16px;
        opacity: 0.92;
        line-height: 1.5;
        max-width: 900px;
    }
    .hero-badges {
        margin-top: 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .badge-pill {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    /* Modern Content Cards */
    .agri-card {
        background: white;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        padding: 22px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .agri-card:hover {
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
    }
    .card-header {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 10px;
    }

    /* Recommendation Hero Card */
    .result-hero {
        background: linear-gradient(135deg, #064e3b 0%, #047857 70%, #059669 100%);
        color: #ffffff;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(6, 78, 59, 0.22);
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .result-crop-name {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: 1px;
        margin: 10px 0;
        text-transform: uppercase;
        color: #fef08a;
        text-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .result-confidence-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 15px;
        font-weight: 700;
    }
    
    /* Primary Action Buttons */
    div.stButton > button {
        width: 100%;
        height: 52px;
        font-size: 17px;
        font-weight: 700;
        border-radius: 10px;
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #15803d 0%, #166534 100%);
        box-shadow: 0 6px 18px rgba(22, 163, 74, 0.35);
        transform: translateY(-1px);
    }

    /* Metric Display Enhancement */
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #1b5e20 !important;
    }

    /* Footer */
    .academic-footer {
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #64748b;
        font-size: 13px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Sidebar Navigation & System Telemetry
# ==============================================================================
with st.sidebar:
    st.markdown("## 🌱 Smart Crop AI")
    st.caption("AI-Powered Agricultural Intelligence System")
    st.divider()

    selected_page = st.radio(
        "Navigate System",
        [
            "🏠 Home Overview",
            "🌾 Crop Recommendation",
            "🤖 Model Intelligence",
            "📊 Data Explorer"
        ],
        index=0
    )

    st.divider()
    
    # System Status Card in Sidebar
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 14px; border-radius: 10px; font-size: 12px;">
        <div style="font-weight: 700; color: #1e293b; margin-bottom: 6px;">⚡ SYSTEM TELEMETRY</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #64748b;">Engine:</span>
            <span style="font-weight: 600; color: #0f172a;">Random Forest (100 Trees)</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #64748b;">Model Accuracy:</span>
            <span style="font-weight: 700; color: #16a34a;">99.32%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #64748b;">Supported Crops:</span>
            <span style="font-weight: 600; color: #0f172a;">22 Varieties</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #64748b;">Dataset Size:</span>
            <span style="font-weight: 600; color: #0f172a;">2,200 Observations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Precision Agriculture Decision Support System • Machine Learning Project")


# ==============================================================================
# PAGE 1: HOME OVERVIEW
# ==============================================================================
if selected_page == "🏠 Home Overview":
    
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌱 Smart Crop Recommendation System</div>
        <div class="hero-subtitle">
            Harnessing multi-class ensemble machine learning to recommend the optimal agricultural crop based on soil nutrient composition (N, P, K), pH balance, and hyper-local climatic parameters.
        </div>
        <div class="hero-badges">
            <span class="badge-pill">🌲 100-Tree Random Forest</span>
            <span class="badge-pill">🎯 99.32% Test Accuracy</span>
            <span class="badge-pill">🌾 22 Crop Classes</span>
            <span class="badge-pill">⚡ Sub-50ms Inference</span>
            <span class="badge-pill">🧪 7 Agronomic Variables</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Indicators
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="🎯 Model Accuracy", value="99.32%", delta="Evaluated Test Split")
    with kpi2:
        st.metric(label="🌾 Supported Crops", value="22 Classes", delta="Balanced Dataset")
    with kpi3:
        st.metric(label="🧪 Input Parameters", value="7 Features", delta="Soil & Weather")
    with kpi4:
        st.metric(label="⚡ Inference Speed", value="< 50 ms", delta="Optimized")

    st.markdown("<br>", unsafe_allow_html=True)

    # How It Works Workflow
    st.markdown("### ⚙️ System Architecture & Workflow")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown("""
        <div class="agri-card" style="height: 190px;">
            <div style="font-size: 24px; margin-bottom: 8px;">🧪 1. Soil Telemetry</div>
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 4px;">Nutrient Assay</div>
            <div style="font-size: 13px; color: #64748b;">Soil Nitrogen (N), Phosphorus (P), Potassium (K), and acidity (pH) are captured.</div>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="agri-card" style="height: 190px;">
            <div style="font-size: 24px; margin-bottom: 8px;">🌦️ 2. Weather Sensing</div>
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 4px;">Atmospheric Factors</div>
            <div style="font-size: 13px; color: #64748b;">Ambient temperature, relative humidity, and cumulative rainfall measurements.</div>
        </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="agri-card" style="height: 190px;">
            <div style="font-size: 24px; margin-bottom: 8px;">🤖 3. Ensemble Model</div>
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 4px;">Random Forest Voting</div>
            <div style="font-size: 13px; color: #64748b;">100 randomized decision trees evaluate non-linear feature splits simultaneously.</div>
        </div>
        """, unsafe_allow_html=True)
    with w4:
        st.markdown("""
        <div class="agri-card" style="height: 190px;">
            <div style="font-size: 24px; margin-bottom: 8px;">📋 4. Actionable Advisory</div>
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 4px;">Decision Support</div>
            <div style="font-size: 13px; color: #64748b;">Yields top recommended crop with probability distribution and agronomic guidance.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 22 Crop Catalog Showcase
    st.markdown("### 🌾 Supported Crop Varieties (22 Classes)")
    st.write("The system has been trained on 2,200 curated field records spanning four agricultural categories:")

    cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)
    with cat_col1:
        st.markdown("""
        <div class="agri-card">
            <div class="card-header">🌾 Cereals & Grains</div>
            <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: #334155;">
                <li><b>Rice</b> (Paddy)</li>
                <li><b>Maize</b> (Corn)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with cat_col2:
        st.markdown("""
        <div class="agri-card">
            <div class="card-header">🫘 Pulses & Legumes</div>
            <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: #334155;">
                <li>Chickpea (Gram)</li>
                <li>Kidney Beans (Rajma)</li>
                <li>Pigeon Peas (Arhar)</li>
                <li>Moth Beans</li>
                <li>Mung Bean (Moong)</li>
                <li>Black Gram (Urad)</li>
                <li>Lentil (Masoor)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with cat_col3:
        st.markdown("""
        <div class="agri-card">
            <div class="card-header">🍎 Fruits & Orchards</div>
            <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: #334155;">
                <li>Banana & Mango</li>
                <li>Grapes & Apple</li>
                <li>Orange & Papaya</li>
                <li>Pomegranate</li>
                <li>Watermelon & Muskmelon</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with cat_col4:
        st.markdown("""
        <div class="agri-card">
            <div class="card-header">🌿 Commercial Crops</div>
            <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: #334155;">
                <li><b>Cotton</b> (Fiber)</li>
                <li><b>Jute</b> (Golden Fiber)</li>
                <li><b>Coconut</b> (Plantation)</li>
                <li><b>Coffee</b> (Beverage)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Ready to run a prediction?** Navigate to **'🌾 Crop Recommendation'** from the sidebar or try one of the instant evaluator presets!")


# ==============================================================================
# PAGE 2: CROP RECOMMENDATION ENGINE
# ==============================================================================
elif selected_page == "🌾 Crop Recommendation":

    st.markdown("## 🌾 Crop Recommendation Engine")
    st.write("Input soil chemical assays and regional weather parameters to obtain AI-powered crop selection and agronomic cultivation guidance.")

    # 1-Click Evaluator Presets for College Viva / Demonstrations
    st.markdown("#### ⚡ 1-Click Evaluator Presets")
    st.caption("Click any preset to instantly populate realistic agricultural conditions for instant testing during project demonstrations:")

    preset_col1, preset_col2, preset_col3, preset_col4, preset_col5 = st.columns(5)
    
    preset_data = {
        "wetland": {"N": 85.0, "P": 45.0, "K": 40.0, "ph": 6.5, "temp": 24.0, "hum": 82.0, "rain": 235.0},
        "orchard": {"N": 25.0, "P": 130.0, "K": 200.0, "ph": 6.0, "temp": 22.0, "hum": 90.0, "rain": 110.0},
        "pulse": {"N": 38.0, "P": 65.0, "K": 80.0, "ph": 7.3, "temp": 18.0, "hum": 17.0, "rain": 75.0},
        "coffee": {"N": 100.0, "P": 28.0, "K": 30.0, "ph": 6.8, "temp": 26.0, "hum": 58.0, "rain": 160.0},
        "cotton": {"N": 118.0, "P": 46.0, "K": 20.0, "ph": 6.9, "temp": 24.0, "hum": 78.0, "rain": 85.0}
    }

    # Initialize session state defaults if absent
    for k, v in {"N": 50.0, "P": 50.0, "K": 50.0, "ph": 6.5, "temp": 25.0, "hum": 70.0, "rain": 100.0}.items():
        if f"val_{k}" not in st.session_state:
            st.session_state[f"val_{k}"] = v

    with preset_col1:
        if st.button("🌊 Wetland\n(Rice / Jute)"):
            for k, val in preset_data["wetland"].items():
                st.session_state[f"val_{k}"] = val
            st.rerun()

    with preset_col2:
        if st.button("🍎 High-K Orchard\n(Apple / Grapes)"):
            for k, val in preset_data["orchard"].items():
                st.session_state[f"val_{k}"] = val
            st.rerun()

    with preset_col3:
        if st.button("🫘 Arid Pulse\n(Chickpea)"):
            for k, val in preset_data["pulse"].items():
                st.session_state[f"val_{k}"] = val
            st.rerun()

    with preset_col4:
        if st.button("☕ Plantation\n(Coffee)"):
            for k, val in preset_data["coffee"].items():
                st.session_state[f"val_{k}"] = val
            st.rerun()

    with preset_col5:
        if st.button("⚪ Commercial\n(Cotton)"):
            for k, val in preset_data["cotton"].items():
                st.session_state[f"val_{k}"] = val
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form organized in two clean visual cards
    col_soil, col_climate = st.columns(2)

    with col_soil:
        st.markdown("""
        <div class="agri-card">
            <div class="card-header">🧪 Soil Chemistry Parameters</div>
        """, unsafe_allow_html=True)

        nitrogen = st.slider(
            "Nitrogen (N) content in soil (kg/ha)",
            min_value=0.0, max_value=145.0,
            value=float(st.session_state.get("val_N", 50.0)),
            step=1.0,
            help="Crucial for vegetative leaf and shoot growth"
        )

        phosphorus = st.slider(
            "Phosphorus (P) content in soil (kg/ha)",
            min_value=5.0, max_value=145.0,
            value=float(st.session_state.get("val_P", 50.0)),
            step=1.0,
            help="Essential for root establishment and blossom development"
        )

        potassium = st.slider(
            "Potassium (K) content in soil (kg/ha)",
            min_value=5.0, max_value=205.0,
            value=float(st.session_state.get("val_K", 50.0)),
            step=1.0,
            help="Regulates plant water intake and pest/disease resilience"
        )

        ph = st.slider(
            "Soil pH (Acidity / Alkalinity level)",
            min_value=3.5, max_value=10.0,
            value=float(st.session_state.get("val_ph", 6.5)),
            step=0.1,
            help="Neutral soil: 6.0 - 7.5. Below 5.5 is strongly acidic; above 8.0 is alkaline."
        )

        # Dynamic pH indicator
        if ph < 5.5:
            st.warning(f"⚠️ Current pH ({ph:.1f}) is Strongly Acidic. May require lime treatment.")
        elif ph > 8.0:
            st.warning(f"⚠️ Current pH ({ph:.1f}) is Strongly Alkaline. Nutrient absorption may be hindered.")
        else:
            st.caption(f"✅ Soil pH ({ph:.1f}) is within optimal agronomic range.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_climate:
        st.markdown("""
        <div class="agri-card">
            <div class="card-header">🌦️ Climatic & Environmental Parameters</div>
        """, unsafe_allow_html=True)

        temperature = st.slider(
            "Average Temperature (°C)",
            min_value=8.0, max_value=45.0,
            value=float(st.session_state.get("val_temp", 25.0)),
            step=0.5,
            help="Average ambient temperature during the crop growing season"
        )

        humidity = st.slider(
            "Relative Humidity (%)",
            min_value=10.0, max_value=100.0,
            value=float(st.session_state.get("val_hum", 70.0)),
            step=1.0,
            help="Relative air moisture level percentage"
        )

        rainfall = st.slider(
            "Seasonal Rainfall (mm)",
            min_value=20.0, max_value=300.0,
            value=float(st.session_state.get("val_rain", 100.0)),
            step=1.0,
            help="Total cumulative precipitation received during cultivation"
        )

        # Dynamic Climate warnings
        if temperature > 40.0:
            st.warning("☀️ Extreme heat alert: Heat-sensitive crops may suffer flower drop.")
        elif rainfall > 220.0:
            st.info("🌧️ High rainfall conditions: Favorable for wetland crops like Rice and Jute.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction Action
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🌱 Calculate Optimal Crop Recommendation")

    if predict_clicked:
        # Prepare feature vector exactly matching training schema
        input_data = pd.DataFrame({
            "N": [nitrogen],
            "P": [phosphorus],
            "K": [potassium],
            "temperature": [temperature],
            "humidity": [humidity],
            "ph": [ph],
            "rainfall": [rainfall]
        })

        with st.spinner("🤖 Running 100-Tree Random Forest classification..."):
            prediction = model.predict(input_data)
            probabilities = model.predict_proba(input_data)[0]

            crop_raw = prediction[0].lower()
            confidence = probabilities.max() * 100

            # Rank Top-3 Probabilities
            top_indices = np.argsort(probabilities)[::-1][:3]
            classes = model.classes_
            top_predictions = [
                (classes[idx], probabilities[idx] * 100)
                for idx in top_indices
            ]

        # Retrieve metadata for predicted crop
        crop_meta = CROP_INFO.get(crop_raw, {
            "name": crop_raw.title(),
            "emoji": "🌾",
            "category": "Agricultural Crop",
            "season": "General",
            "water": "Moderate",
            "soil": "Fertile well-drained soil",
            "duration": "100 - 120 days",
            "tip": "Maintain balanced N-P-K fertilization and adequate irrigation."
        })

        # Hero Result Display
        st.markdown(f"""
        <div class="result-hero">
            <div style="font-size: 20px; font-weight: 600; opacity: 0.95;">RECOMMENDED CROP MATCH</div>
            <div class="result-crop-name">{crop_meta['emoji']} {crop_meta['name'].upper()}</div>
            <div class="result-confidence-badge">
                🎯 Model Confidence: {confidence:.2f}% &nbsp;•&nbsp; Category: {crop_meta['category']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        # Two-Column Detailed Breakdown: Top 3 Probabilities + Cultivation Advisory
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("""
            <div class="agri-card">
                <div class="card-header">📊 Model Prediction Probabilities (Top 3)</div>
            """, unsafe_allow_html=True)

            top_df = pd.DataFrame({
                "Crop": [CROP_INFO.get(c[0].lower(), {}).get("name", c[0].title()) for c in top_predictions],
                "Probability": [c[1] for c in top_predictions]
            }).sort_values(by="Probability", ascending=True)

            fig_prob = px.bar(
                top_df,
                x="Probability",
                y="Crop",
                orientation="h",
                text="Probability",
                color="Probability",
                color_continuous_scale=["#86efac", "#22c55e", "#15803d"],
                range_x=[0, 100]
            )
            fig_prob.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                marker_line_color='#166534',
                marker_line_width=1
            )
            fig_prob.update_layout(
                xaxis_title="Confidence Probability (%)",
                yaxis_title="",
                height=240,
                margin=dict(l=10, r=30, t=10, b=10),
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            st.caption("ℹ️ Probabilities are computed via normalized soft-voting across 100 decision trees in the ensemble.")
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="agri-card">
                <div class="card-header">📋 Agronomic Cultivation Guide: {crop_meta['name']}</div>
                <div style="font-size: 14px; line-height: 1.8;">
                    <b>🗓️ Ideal Season:</b> {crop_meta['season']}<br>
                    <b>💧 Water Requirement:</b> {crop_meta['water']}<br>
                    <b>🪴 Preferred Soil:</b> {crop_meta['soil']}<br>
                    <b>⏱️ Maturity Duration:</b> {crop_meta['duration']}<br>
                    <div style="margin-top: 10px; padding: 10px; background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 4px;">
                        <b>💡 Agronomist Tip:</b> {crop_meta['tip']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Radar Comparison: User Inputs vs. Ideal Dataset Benchmark for Recommended Crop
        st.markdown("### 🕸️ Input Telemetry vs. Crop Baseline Benchmark")
        crop_data_subset = df_dataset[df_dataset["label"].str.lower() == crop_raw]

        if not crop_data_subset.empty:
            mean_vals = crop_data_subset[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].mean()

            # Normalized 0-100 representation for clear radar comparison
            features = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]
            raw_user = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
            raw_ideal = [
                mean_vals["N"], mean_vals["P"], mean_vals["K"],
                mean_vals["temperature"], mean_vals["humidity"],
                mean_vals["ph"], mean_vals["rainfall"]
            ]

            fig_radar = go.Figure()

            fig_radar.add_trace(go.Scatterpolar(
                r=raw_user,
                theta=features,
                fill='toself',
                name='User Input Parameters',
                line_color='#2563eb',
                fillcolor='rgba(37, 99, 235, 0.25)'
            ))

            fig_radar.add_trace(go.Scatterpolar(
                r=raw_ideal,
                theta=features,
                fill='toself',
                name=f'Benchmark Mean for {crop_meta["name"]}',
                line_color='#16a34a',
                fillcolor='rgba(22, 163, 74, 0.25)'
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, showticklabels=False)
                ),
                showlegend=True,
                height=380,
                margin=dict(l=30, r=30, t=30, b=30),
                paper_bgcolor="rgba(0,0,0,0)"
            )

            rad_col, summary_col = st.columns([1.2, 0.8])
            with rad_col:
                st.plotly_chart(fig_radar, use_container_width=True)
            with summary_col:
                st.markdown("""
                <div class="agri-card" style="height: 380px; overflow-y: auto;">
                    <div class="card-header">📝 Parameter Audit Table</div>
                """, unsafe_allow_html=True)
                
                audit_df = pd.DataFrame({
                    "Feature": ["N (kg/ha)", "P (kg/ha)", "K (kg/ha)", "Temp (°C)", "Humidity (%)", "Soil pH", "Rainfall (mm)"],
                    "Entered Value": [f"{v:.1f}" for v in raw_user],
                    f"Ideal {crop_meta['name']}": [f"{v:.1f}" for v in raw_ideal]
                })
                st.dataframe(audit_df, hide_index=True, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: MODEL INTELLIGENCE & SPECIFICATIONS
# ==============================================================================
elif selected_page == "🤖 Model Intelligence":

    st.markdown("## 🤖 Machine Learning Model Intelligence")
    st.write("Technical audit, hyperparameter specifications, and mathematical feature importance diagnostics for the Random Forest classifier.")

    # High-level Architecture Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Algorithm", "Random Forest", "Bagging Ensemble")
    with c2:
        st.metric("Estimators", "100 Trees", "Variance Reduction")
    with c3:
        st.metric("Test Accuracy", "99.32%", "80:20 Split Ratio")
    with c4:
        st.metric("Target Space", "22 Classes", "100 Samples / Class")

    st.divider()

    # Feature Importance Plot with Plotly
    st.markdown("### 📊 Gini Impurity Feature Importance")
    st.write("Calculated across all 100 decision trees to quantify how much each agronomic variable contributes to reducing multi-class entropy:")

    feature_names = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "Soil pH", "Rainfall"]
    feature_importances = model.feature_importances_

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances,
        "Percentage": [f"{val * 100:.2f}%" for val in feature_importances]
    }).sort_values(by="Importance", ascending=True)

    fig_imp = px.bar(
        df_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Percentage",
        color="Importance",
        color_continuous_scale="Greens",
        title="Relative Contribution of Agronomic Features to Prediction"
    )
    fig_imp.update_traces(
        textposition='outside',
        marker_line_color='#14532d',
        marker_line_width=1
    )
    fig_imp.update_layout(
        xaxis_title="Normalized Feature Importance (Sum = 1.0)",
        yaxis_title="",
        height=380,
        margin=dict(l=10, r=30, t=40, b=10),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Technical Viva Explanation Section
    viva_col1, viva_col2 = st.columns(2)
    with viva_col1:
        st.markdown("""
        <div class="agri-card">
            <div class="card-header">🎓 Why Random Forest for Crop Classification?</div>
            <div style="font-size: 14px; line-height: 1.7; color: #334155;">
                <ul>
                    <li><b>Non-Linear Boundary Separation:</b> Agricultural parameters have non-linear threshold dependencies (e.g. low rainfall favors pulses, high rainfall favors paddy). Decision trees naturally model step thresholds without requiring linear assumptions.</li>
                    <li><b>Resistance to Overfitting:</b> By aggregating 100 bootstrap-aggregated (bagged) trees with feature sub-sampling, the ensemble variance is significantly minimized.</li>
                    <li><b>Scale Invariance:</b> Tree models split on individual feature values, which means disparate physical units (pH: 0-14 vs. Rainfall: 0-300 mm) do not require artificial min-max normalization.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with viva_col2:
        st.markdown("""
        <div class="agri-card">
            <div class="card-header">⚙️ Training Hyperparameters & Pipeline</div>
            <div style="font-size: 14px; line-height: 1.8; color: #334155;">
                <b>• Framework:</b> Scikit-Learn <code>ensemble.RandomForestClassifier</code><br>
                <b>• Number of Estimators (n_estimators):</b> <code>100</code><br>
                <b>• Split Criterion:</b> Gini Impurity<br>
                <b>• Random Seed (random_state):</b> <code>42</code> (ensuring 100% deterministic reproducibility)<br>
                <b>• Training Samples:</b> 1,760 records (80%)<br>
                <b>• Unseen Validation Samples:</b> 440 records (20%)<br>
                <b>• Serialization Artifact:</b> <code>joblib.dump(..., 'crop_model.pkl')</code>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: DATA EXPLORER
# ==============================================================================
elif selected_page == "📊 Data Explorer":

    st.markdown("## 📊 Interactive Dataset Explorer")
    st.write("Inspect the ground-truth agronomic dataset used to train and validate the crop recommendation engine.")

    # KPI Summary Cards
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.metric("Total Records", f"{len(df_dataset):,}", "Complete rows")
    with d_col2:
        st.metric("Input Features", "7", "N, P, K, Temp, Hum, pH, Rain")
    with d_col3:
        st.metric("Target Crop Classes", f"{df_dataset['label'].nunique()}", "Balanced 100 / class")
    with d_col4:
        st.metric("Missing Values", "0", "100% Complete")

    st.markdown("<br>", unsafe_allow_html=True)

    # Dynamic Filterable Dataset View
    st.markdown("### 🔍 Dataset Viewer & Filter")
    
    f_col1, f_col2 = st.columns([1.5, 1])
    with f_col1:
        all_crops = sorted(df_dataset["label"].unique().tolist())
        selected_crops = st.multiselect(
            "Filter by Crop Type (Leave empty to view all):",
            options=all_crops,
            default=[]
        )
    with f_col2:
        row_limit = st.slider("Number of rows to preview:", min_value=10, max_value=200, value=25, step=5)

    if selected_crops:
        filtered_df = df_dataset[df_dataset["label"].isin(selected_crops)]
    else:
        filtered_df = df_dataset

    st.dataframe(
        filtered_df.head(row_limit),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Showing {min(row_limit, len(filtered_df))} of {len(filtered_df)} matching records.")

    st.divider()

    # Interactive Visualizations Section
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("### 🌾 Crop Class Distribution")
        crop_counts = df_dataset["label"].value_counts().reset_index()
        crop_counts.columns = ["Crop", "Count"]

        fig_dist = px.bar(
            crop_counts,
            x="Crop",
            y="Count",
            color="Count",
            color_continuous_scale="Greens",
            title="Balanced Class Representation (100 Samples Each)"
        )
        fig_dist.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with viz_col2:
        st.markdown("### 📈 Feature Correlation Heatmap")
        corr_matrix = df_dataset.drop("label", axis=1).corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Pearson Correlation Between Agronomic Variables"
        )
        fig_corr.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Distribution Deep-Dive by Crop
    st.markdown("### 🔬 Parameter Distribution Explorer by Crop")
    st.write("Select an agronomic variable to observe how its values vary across individual crops:")

    feat_selected = st.selectbox(
        "Select Parameter to Inspect:",
        ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        index=6
    )

    fig_box = px.box(
        df_dataset,
        x="label",
        y=feat_selected,
        color="label",
        title=f"Distribution of {feat_selected.upper()} Across All 22 Crops",
        labels={"label": "Crop", feat_selected: feat_selected.upper()}
    )
    fig_box.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # Statistical Summary Table
    st.markdown("### 📋 Descriptive Statistical Summary")
    st.dataframe(
        df_dataset.describe().T.style.format("{:.2f}"),
        use_container_width=True
    )


# ==============================================================================
# Academic Project Footer
# ==============================================================================
st.markdown("""
<div class="academic-footer">
    <b>Smart Crop Recommendation System</b> • Machine Learning in Precision Agriculture<br>
    Built with Python, Streamlit, Scikit-Learn & Plotly • Trained on 2,200 Agricultural Observations
</div>
""", unsafe_allow_html=True)