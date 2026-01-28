import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF

# Import backend modules
from config import Config
from DataManager import DataManager
from DataPreprocessor import Preprocessor
from ModelManager import ModelManager

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AgroSpectral Analyzer", layout="wide", page_icon="🌱")

# --- PDF GENERATION CLASS ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Soil Analysis Report (Hyperspectral)', 0, 1, 'C')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, body)
        self.ln()

# --- SETUP FUNCTIONS (CACHED) ---
@st.cache_resource
def load_system():
    cfg = Config()
    dm = DataManager(cfg)
    pp = Preprocessor(cfg)
    mm = ModelManager(cfg)
    
    # Attempt automatic model loading
    default_model_path = os.path.join(cfg.SUBMISSION_DIR, 'Models', 'best_rf_model.pkl')
    if not os.path.exists(default_model_path):
        default_model_path = 'best_rf_model.pkl'

    if os.path.exists(default_model_path):
        try:
            mm.load_models(default_model_path)
            print(f"Automatically loaded model: {default_model_path}")
        except Exception as e:
            print(f"Error loading default model: {e}")

    return cfg, dm, pp, mm

# --- HELPER FUNCTIONS ---

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving temporary file: {e}")
        return None

def create_pdf(dataframe):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Sample Analysis Results:", ln=1, align='L')
    pdf.ln(5)
    
    cols = dataframe.columns.tolist()
    header = " | ".join(cols)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(0, 10, txt=header, ln=1)
    
    pdf.set_font("Arial", '', 8)
    for index, row in dataframe.iterrows():
        row_values = []
        for val in row.values:
            if isinstance(val, (float, np.floating)):
                row_values.append(f"{val:.2f}")
            else:
                row_values.append(str(val))
        row_str = " | ".join(row_values)
        pdf.cell(0, 8, txt=row_str, ln=1)
        
    return pdf.output(dest='S').encode('latin-1')

def process_analysis(uploaded_files, dm, pp, mm, include_higher_derivs=False):
    results_list = []
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        temp_path = save_uploaded_file(uploaded_file)
        if temp_path:
            try:
                img = dm.load_image_from_path(temp_path)
                if img is not None:
                    features = pp.process_image(img, include_higher_derivs=include_higher_derivs)
                    if features is not None:
                        features_reshaped = features.reshape(1, -1)
                        pred_df = mm.predict(features_reshaped)
                        res_dict = pred_df.iloc[0].to_dict()
                        res_dict['File'] = uploaded_file.name
                        results_list.append(res_dict)
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")
            finally:
                os.remove(temp_path)
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    return pd.DataFrame(results_list)

# --- MAIN APPLICATION ---
def main():
    cfg, dm, pp, mm = load_system()
    
    st.sidebar.title("AgroSpectral Analyzer")
    role = st.sidebar.radio("Select Mode:", ["Farmer", "Scientist"])
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Config Model Type: **{cfg.MODEL_TYPE.upper()}**")
    
    if mm.models:
        st.sidebar.success("✅ Model Loaded")
    else:
        st.sidebar.error("❌ No Model Loaded")

    # --- ROLE: FARMER ---
    if role == "Farmer":
        st.header("🚜 Soil Analysis Panel")
        uploaded_files = st.file_uploader("Upload samples (.npz)", type=["npz"], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Run Analysis"):
                if not mm.models:
                    st.error("Error: Model not loaded.")
                else:
                    with st.spinner("Processing images..."):
                        # Assuming False for best_rf_model unless specified otherwise
                        results_df = process_analysis(uploaded_files, dm, pp, mm, include_higher_derivs=False)
                    
                    if not results_df.empty:
                        st.success("Analysis Complete!")
                        st.dataframe(results_df)
                        
                        col1, col2 = st.columns(2)
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        col1.download_button("Download CSV", csv, "results.csv", "text/csv")
                        try:
                            pdf_data = create_pdf(results_df)
                            col2.download_button("Download PDF", pdf_data, "report.pdf", "application/pdf")
                        except Exception as e:
                            col2.warning(f"PDF Error: {e}")

    # --- ROLE: SCIENTIST ---
    elif role == "Scientist":
        st.header("🔬 Scientific Panel")
        
        tabs = st.tabs(["Model Management", "Data Exploration", "Feature Selection", "Batch Submission"])
        
        # TAB 1: MODEL
        with tabs[0]:
            st.subheader("Model Selection")
            models_dir = os.path.join(cfg.SUBMISSION_DIR, 'Models')
            if not os.path.exists(models_dir): models_dir = "."
            available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and not f.endswith('_selector.pkl')]
            
            selected_model = st.selectbox("Available models:", available_models if available_models else ["No models found"])
            
            if st.button("Load Selected Model"):
                path = os.path.join(models_dir, selected_model)
                try:
                    mm.load_models(path)
                    st.toast(f"Loaded model: {selected_model}", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
            
            st.markdown("---")
            if mm.selector:
                if isinstance(mm.selector, dict):
                    st.success("Detected and loaded Feature Selector Dictionary (specific per target).")
                else:
                    st.success("Detected and loaded Single Feature Selector.")
            else:
                st.warning("No active Feature Selector.")

        # TAB 2: EXPLORATION
        with tabs[1]:
            st.subheader("Spectrum Preview")
            exp_file = st.file_uploader("Upload single .npz file", type=["npz"])
            if exp_file:
                temp_path = save_uploaded_file(exp_file)
                if temp_path:
                    img = dm.load_image_from_path(temp_path)
                    os.remove(temp_path)
                    if img is not None:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        mean_spectrum = np.mean(img.data, axis=(1, 2))
                        ax.plot(mean_spectrum, color='blue')
                        ax.set_title("Mean Spectrum")
                        ax.set_xlabel("Band Index")
                        ax.set_ylabel("Reflectance")
                        st.pyplot(fig)

        # TAB 3: FEATURE SELECTION
        with tabs[2]:
            st.subheader("Feature Importance")
            if mm.selector is None:
                st.info("No feature selection active.")
            else:
                selectors_to_show = {}
                
                if isinstance(mm.selector, dict):
                    st.info("Displaying ranking for each target parameter separately.")
                    selectors_to_show = mm.selector
                else:
                    selectors_to_show = {'Global Selector': mm.selector}
                
                for key, sel_obj in selectors_to_show.items():
                    with st.expander(f"Parameter: {key}", expanded=True):
                        try:
                            ranking = None
                            n_features = "Unknown"

                            # Check if Wrapper or RFE
                            if hasattr(sel_obj, 'ranking_'): # RFE object
                                ranking = sel_obj.ranking_
                                n_features = getattr(sel_obj, 'n_features_to_select', 'N/A')
                            elif hasattr(sel_obj, 'selector_'): # Custom Wrapper
                                ranking = sel_obj.selector_.ranking_
                                n_features = getattr(sel_obj.selector_, 'n_features_to_select', 'N/A')
                            
                            st.write(f"Reduced to: **{n_features}** features.")

                            if ranking is not None:
                                fig_r, ax_r = plt.subplots(figsize=(10, 3))
                                ax_r.bar(range(len(ranking)), ranking, color='purple')
                                ax_r.set_title(f"Feature Ranking for {key}")
                                ax_r.set_xlabel("Feature Index")
                                ax_r.set_ylabel("Rank")
                                ax_r.axhline(y=1.5, color='r', linestyle='--')
                                st.pyplot(fig_r)
                        except Exception as e:
                            st.warning(f"Could not visualize {key}: {e}")

        # TAB 4: SUBMISSION
        with tabs[3]:
            st.subheader("Batch Submission")
            if st.button("Generate Submission CSV"):
                if not mm.models:
                    st.error("Model not loaded!")
                else:
                    try:
                        test_ids = dm.get_data_indices(cfg.TEST_DATA_PATH)
                        X_test, valid_ids = pp.process_dataset(dm, test_ids, cfg.TEST_DATA_PATH, include_higher_derivs=False)
                        preds_df = mm.predict(X_test)
                        dm.save_submission(preds_df.values, filename="submission_app.csv")
                        with open("submission_app.csv", "rb") as f:
                            st.download_button("Download CSV", f, "submission.csv")
                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()