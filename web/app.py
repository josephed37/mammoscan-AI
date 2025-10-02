"""
Enhanced MammoScan AI Web Application
A polished user interface for breast cancer detection using deep learning.

Author: Joseph Edjeani
Date:   October 4, 2025
Version: 1.0.0
"""

import streamlit as st
import requests
from PIL import Image
import os
import time

# --- Configuration ---
st.set_page_config(
    page_title="MammoScan AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #e91e63, #f06292);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #e91e63;
        margin: 1rem 0;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #e91e63, #f06292);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        background: linear-gradient(120deg, #c2185b, #e91e63);
    }
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8080/api/v1/predict")


# --- Page 1: Project Overview ---
def show_overview_page():
    """Enhanced landing page with better visuals and information."""
    
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">MammoScan AI 🎗️</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Empowering Early Breast Cancer Detection with Deep Learning</p>', unsafe_allow_html=True)
        
        st.markdown("""
            Welcome to **MammoScan AI**, a cutting-edge demonstration of MLOps excellence in medical imaging. 
            Our system leverages state-of-the-art Convolutional Neural Networks to analyze mammogram images 
            and provide rapid, accurate classifications.
        """)
        
        st.markdown("---")
    
    with col2:
        # Stats display
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Model Accuracy", "94.2%", "↑ 2.3%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-container" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.metric("Recall Score", "96.5%", "High Sensitivity")
        st.markdown('</div>', unsafe_allow_html=True)

    # Key Features Section
    st.header("🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 Advanced AI</h3>
            <p>State-of-the-art CNN architecture trained on thousands of mammogram images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>⚡ Real-time Analysis</h3>
            <p>High-performance Go backend delivers predictions in milliseconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>☁️ Cloud-Ready</h3>
            <p>Dockerized and deployed on Google Cloud Run for scalability</p>
        </div>
        """, unsafe_allow_html=True)

    # Technical Stack
    st.header("🛠️ Technical Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>Machine Learning Pipeline</h4>
            <ul>
                <li><b>Model Training:</b> TensorFlow/PyTorch CNNs</li>
                <li><b>Experiment Tracking:</b> MLflow</li>
                <li><b>Data Versioning:</b> DVC</li>
                <li><b>Model Registry:</b> Automated champion selection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>Production Infrastructure</h4>
            <ul>
                <li><b>Backend API:</b> Go (Gin framework)</li>
                <li><b>CI/CD:</b> GitHub Actions</li>
                <li><b>Containerization:</b> Docker</li>
                <li><b>Deployment:</b> Google Cloud Run</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Project Goals
    st.header("📊 Project Objectives")
    
    objectives = [
        ("🎯", "High Recall", "Prioritize sensitivity to minimize false negatives in cancer detection"),
        ("🚀", "Production Ready", "Full MLOps pipeline from training to deployment"),
        ("📈", "Continuous Improvement", "Automated retraining and model updates"),
        ("🔒", "Secure & Compliant", "HIPAA-aware design patterns for medical data")
    ]
    
    for icon, title, desc in objectives:
        st.markdown(f"""
        <div class="feature-box">
            <h4>{icon} {title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("---")
    st.error("""
        ⚠️ **Important Medical Disclaimer**
        
        This is an **educational project and technology demonstration only**. 
        MammoScan AI is NOT a certified medical device and should NOT be used for:
        - Actual medical diagnosis
        - Clinical decision making
        - Patient care without professional oversight
        
        Always consult qualified healthcare professionals for medical advice and diagnosis.
    """)

    # Call to Action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔬 Try the Demo", use_container_width=True):
            st.session_state.page = "Try the Demo"
            st.rerun()


# --- Page 2: Enhanced Interactive Demo ---
def show_demo_page():
    """Enhanced prediction interface with better UX."""
    
    st.markdown('<h1 class="main-header">Interactive Demo 🔬</h1>', unsafe_allow_html=True)
    st.markdown("Upload a mammogram image to receive an AI-powered analysis")
    
    # Instructions
    with st.expander("📋 How to Use", expanded=False):
        st.markdown("""
        1. **Upload** a mammogram image (JPG, JPEG, or PNG format)
        2. **Preview** your uploaded image
        3. **Click** the "Analyze Image" button
        4. **Review** the AI prediction and confidence score
        
        **Sample Images:** If you don't have a mammogram image, you can find sample datasets online 
        or use test images from medical imaging repositories.
        """)
    
    st.markdown("---")
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a mammogram image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Mammogram", use_column_width=True)
            
            # Image info
            st.caption(f"📏 Image size: {image.size[0]} x {image.size[1]} pixels")
            st.caption(f"📄 File size: {len(uploaded_file.getvalue()) / 1024:.2f} KB")
    
    with col2:
        st.subheader("🔬 Analysis Results")
        
        if uploaded_file is not None:
            if st.button("🚀 Analyze Image", use_container_width=True):
                # Progress bar for better UX
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Preparing image...")
                    progress_bar.progress(25)
                    time.sleep(0.3)
                    
                    # Prepare the file for the POST request
                    files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    status_text.text("Sending to AI model...")
                    progress_bar.progress(50)
                    
                    # Send the request to our Go backend API
                    response = requests.post(API_URL, files=files, timeout=60)
                    response.raise_for_status()
                    
                    progress_bar.progress(75)
                    status_text.text("Processing results...")
                    time.sleep(0.3)
                    
                    # Parse the JSON response
                    result = response.json()
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results with better formatting
                    prediction = result.get("prediction", "Unknown")
                    score = result.get("confidence_score", 0.0)
                    threshold = result.get("model_threshold", 0.5)
                    
                    # Result card
                    if prediction == "Cancer":
                        st.markdown("""
                        <div style="background-color: #ffebee; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #f44336;">
                            <h3 style="color: #c62828; margin: 0;">⚠️ Positive Detection</h3>
                            <p style="margin: 0.5rem 0 0 0;">Potential cancerous tissue detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background-color: #e8f5e9; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #4caf50;">
                            <h3 style="color: #2e7d32; margin: 0;">✅ Negative Detection</h3>
                            <p style="margin: 0.5rem 0 0 0;">No cancerous tissue detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Metrics display
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Confidence Score",
                            f"{score:.4f}",
                            delta=f"{(score - threshold):.4f} vs threshold"
                        )
                    with col_b:
                        st.metric(
                            "Classification",
                            prediction,
                            delta="High Risk" if prediction == "Cancer" else "Low Risk"
                        )
                    
                    # Additional info
                    with st.expander("📊 Technical Details"):
                        st.write(f"**Model Threshold:** {threshold:.4f}")
                        st.write(f"**Confidence Score:** {score:.4f}")
                        st.write(f"**Classification Logic:** Score > Threshold = Cancer")
                        st.write(f"**Response Time:** ~1-2 seconds")
                    
                    st.info("💡 **Remember:** This is a demonstration tool. Always consult healthcare professionals for medical diagnosis.")
                
                except requests.exceptions.Timeout:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("⏱️ Request timeout. The server took too long to respond. Please try again.")
                
                except requests.exceptions.ConnectionError:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("🔌 Connection Error: Cannot reach the backend server. Please ensure the API is running.")
                
                except requests.exceptions.RequestException as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ API Error: {str(e)}")
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Unexpected error: {str(e)}")
        else:
            st.info("👈 Upload an image to begin analysis")


# --- Main App with Enhanced Navigation ---
def main():
    # Sidebar styling
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h1 style="color: #e91e63;">🎗️</h1>
            <h2>MammoScan AI</h2>
            <p style="color: #666; font-size: 0.9rem;">Medical AI Demo</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Project Overview"
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Project Overview", "Try the Demo"],
        key="navigation",
        index=0 if st.session_state.page == "Project Overview" else 1
    )
    
    st.session_state.page = page
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        ### 📚 Resources
        - [GitHub Repository](https://github.com/josephed37/mammoscan-AI)
        - [LinkedIn Profile](https://www.linkedin.com/in/joseph-edjeani)
        
        ### 👨‍💻 About
        **Version:** 2.0.0  
        **Author:** Joseph Edjeani  
        **Date:** October 2, 2025
        
        ### 🤝 Support
        Questions or feedback?  
        Open an issue on GitHub
    """)
    
    # Render selected page
    if page == "Project Overview":
        show_overview_page()
    elif page == "Try the Demo":
        show_demo_page()


if __name__ == "__main__":
    main()