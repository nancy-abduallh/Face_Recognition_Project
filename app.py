import torch
import torch.nn as nn
import torchvision.models as models_torch
import torchvision.transforms as transforms
from torchvision.models import GoogLeNet_Weights
from PIL import Image
import numpy as np
import streamlit as st
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="CelebrityVision AI • Face Recognition",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InceptionV1(nn.Module):
    def __init__(self, num_classes=14):
        super(InceptionV1, self).__init__()
        self.backbone = models_torch.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class CelebrityFaceRecognizer:
    def __init__(self, model_path, class_names):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        self.model = InceptionV1(num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded on {self.device}")
        print(f"Classes: {self.class_names}")
    
    def predict(self, image_path, top_k=3, confidence_threshold=0.1):
        """Predict celebrity from image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, k=top_k)
            
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            
            predictions = []
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                predictions.append({
                    'rank': i + 1,
                    'name': self.class_names[idx],
                    'confidence': float(prob) * 100,
                    'is_likely': float(prob) >= confidence_threshold
                })
            
            max_confidence = float(top_probs[0])
            is_unknown = max_confidence < confidence_threshold
            
            return {
                'predictions': predictions,
                'is_unknown': is_unknown,
                'top_prediction': predictions[0] if predictions else None,
                'original_image': image
            }
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None

st.markdown("""
<style>

    .stApp {
        background: linear-gradient(135deg,
            #FFDCDC 0%,
            #FFF2EB 33%,
            #FFE8CD 66%,
            #FFD6BA 100%
        ) !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(8px);
        border-right: 1px solid rgba(0, 0, 0, 0.05);
    }

    .main .block-container {
        background: #FFFFFF !important;
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 15px 30px rgba(0,0,0,0.08);
    }

    h1, h2, h3, h4 {
        color: #333 !important;
        font-weight: 800 !important;
    }

    .stFileUploader > div > div {
        border: 2px dashed #FFB39F !important;
        border-radius: 12px !important;
        background: rgba(255,200,180,0.25) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #FFE8CD, #FFD6BA) !important;
        color: #333 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 2rem !important;
        font-weight: 600 !important;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FFDCDC, #FFE8CD, #FFD6BA) !important;
    }

    .prediction-card {
        background: linear-gradient(135deg,
            #FFF2EB 0%,
            #FFE8CD 100%
        ) !important;
        color: #333 !important;
        padding: 1.3rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        transition: 0.3s ease;
    }

    .prediction-card:hover {
        transform: translateY(-4px);
    }

    [data-testid="stMetricValue"] {
        color: #444 !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: #FFE8CD !important;
        border-radius: 10px 10px 0 0;
        padding: 0.6rem 2rem !important;
        color: #333 !important;
        font-weight: 600 !important;
    }

</style>
""", unsafe_allow_html=True)


def main():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🌟 CelebrityVision AI</h1>
            <p style='color: #666; font-size: 1.2rem;'>Advanced Face Recognition System</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: #667eea;'>⚙️ Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: white; margin-bottom: 1rem;">⚙️ Model Settings</h3>', unsafe_allow_html=True)
            
            confidence_threshold = st.slider(
                "**Confidence Threshold**",
                min_value=0,
                max_value=100,
                value=10,
                help="Minimum confidence percentage to accept prediction",
                key="confidence_slider"
            ) / 100
            
            top_k = st.slider(
                "**Top Predictions**",
                min_value=1,
                max_value=5,
                value=3,
                key="topk_slider"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
       
        
        
        st.markdown('<div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); padding: 1.5rem; border-radius: 15px; margin-top: 1.5rem;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: white;">📊 Model Performance</h3>', unsafe_allow_html=True)
        st.metric("**Accuracy**", "98.66%", "1.34%")
        st.metric("**Classes**", "14", "Fixed")
        st.metric("**Best For**", "All Faces", "-")
        st.markdown('</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📤 Upload & Analyze", "📊 Visualization"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                       padding: 2rem; border-radius: 20px; margin-bottom: 2rem;'>
                <h2>📤 Upload Image</h2>
                <p style='color: #666;'>Upload a clear image of a celebrity face</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                " ",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="📷 Best results with front-facing, well-lit portraits",
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.markdown('<div style="text-align: center; margin-top: 2rem;">', unsafe_allow_html=True)
                image = Image.open(temp_path)
                st.image(image, caption="🎨 Uploaded Image", width=350)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if uploaded_file:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%); 
                           padding: 2rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
                    <h2>🎯 Analysis Results</h2>
                </div>
                """, unsafe_allow_html=True)
                
                @st.cache_resource
                def load_recognizer():
                    if os.path.exists("class_names.pkl"):
                        with open("class_names.pkl", "rb") as f:
                            class_names = pickle.load(f)
                    else:
                        class_names = [
                            "Scarlett Johansson", "Megan Fox", "Natalie Portman",
                            "Jennifer Lawrence", "Denzel Washington", "Hugh Jackman",
                            "Tom Hanks", "Kate Winslet", "Leonardo DiCaprio",
                            "Angelina Jolie", "Will Smith", "Brad Pitt",
                            "Sandra Bullock", "Tom Cruise"
                        ]
                    
                    recognizer = CelebrityFaceRecognizer(
                        model_path="models/inceptionv1_best.pth",  
                        class_names=class_names
                    )
                    return recognizer
                
                try:
                    recognizer = load_recognizer()
                    
                    with st.spinner("🔍 **Analyzing facial features...**"):
                        result = recognizer.predict(
                            temp_path, 
                            top_k=top_k,
                            confidence_threshold=confidence_threshold
                        )
                    
                    if result:
                        if result['is_unknown']:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #FF6B6B 0%, #C44569 100%); 
                                       padding: 1.5rem; border-radius: 15px; color: white; margin: 2rem 0;'>
                                <h3>❌ Unknown Person Detected</h3>
                                <p>The model is not confident about this prediction.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            top_pred = result['top_prediction']
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #06D6A0 0%, #118AB2 100%); 
                                       padding: 2rem; border-radius: 15px; color: white; margin: 2rem 0; text-align: center;'>
                                <h2 style='font-size: 2rem;'>✅ Match Found!</h2>
                                <h1 style='font-size: 2.5rem; margin: 1rem 0;'>{top_pred['name']}</h1>
                                <p style='font-size: 1.2rem;'>Confidence Score</p>
                                <h2 style='font-size: 3rem;'>{top_pred['confidence']:.2f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("### 📊 Detailed Analysis")
                        for pred in result['predictions']:
                            color = "#06D6A0" if pred['is_likely'] else "#FF6B6B"
                            st.markdown(f"""
                            <div class='prediction-card' style='border-left: 5px solid {color};'>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <h3 style='margin: 0;'>#{pred['rank']} {pred['name']}</h3>
                                    <span style='font-size: 1.5rem; font-weight: bold;'>{pred['confidence']:.2f}%</span>
                                </div>
                                <div style='margin-top: 1rem;'>
                                    <div style='height: 8px; background: rgba(255,255,255,0.2); border-radius: 4px;'>
                                        <div style='height: 100%; width: {pred['confidence']}%; background: {color}; border-radius: 4px;'></div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                st.markdown("""
                <div style='text-align: center; padding: 4rem 2rem; color: #666;'>
                    <h2 style='font-size: 2rem;'>👈 Upload an Image</h2>
                    <p style='font-size: 1.2rem;'>Upload a celebrity photo to start analysis</p>
                    <div style='margin-top: 3rem;'>
                        <p style='font-size: 1rem; color: #888;'>💡 <strong>Tips for best results:</strong></p>
                        <p>• Use clear, front-facing photos</p>
                        <p>• Ensure good lighting</p>
                        <p>• Avoid group photos</p>
                        <p>• High resolution recommended</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        if uploaded_file and 'result' in locals():
            st.markdown("### 📈 Visual Analytics")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                names = [p['name'] for p in result['predictions']]
                confidences = [p['confidence'] for p in result['predictions']]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=confidences,
                        y=names,
                        orientation='h',
                        marker=dict(
                            color=confidences,
                            colorscale='Viridis',
                            showscale=True
                        )
                    )
                ])
                
                fig.update_layout(
                    title="Confidence Scores",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Celebrity",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                if len(result['predictions']) >= 3:
                    top_names = [p['name'] for p in result['predictions'][:3]]
                    top_conf = [p['confidence'] for p in result['predictions'][:3]]
                    
                    fig2 = go.Figure(data=[go.Pie(
                        labels=top_names,
                        values=top_conf,
                        hole=.3,
                        marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                    )])
                    
                    fig2.update_layout(
                        title="Top 3 Predictions",
                        height=400,
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
    
    # with tab3:
    #     st.markdown("""
    #     <div style='padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
    #             border-radius: 20px;'>
    #         <h2>🤖 About CelebrityVision AI</h2>
            
    #         <div style='margin: 2rem 0;'>
    #             <h3>✨ Features</h3>
    #             <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;'>
    #                 <div style='background: white; padding: 1rem; border-radius: 10px;'>
    #                     <h4>🎯 High Accuracy</h4>
    #                     <p>98.66% accuracy on celebrity dataset</p>
    #                 </div>
    #                 <div style='background: white; padding: 1rem; border-radius: 10px;'>
    #                     <h4>⚡ Fast Inference</h4>
    #                     <p>Real-time processing capabilities</p>
    #                 </div>
    #                 <div style='background: white; padding: 1rem; border-radius: 10px;'>
    #                     <h4>🔒 Privacy Focused</h4>
    #                     <p>No data storage, local processing</p>
    #                 </div>
    #                 <div style='background: white; padding: 1rem; border-radius: 10px;'>
    #                     <h4>🎨 Modern UI</h4>
    #                     <p>Beautiful, intuitive interface</p>
    #                 </div>
    #             </div>
    #         </div>
            
    #         <div style='margin: 2rem 0;'>
    #             <h3>🛠️ Technology Stack</h3>
    #             <div style='display: flex; gap: 1rem; margin-top: 1rem;'>
    #                 <span style='background: #FF6B6B; color: white; padding: 0.5rem 1rem; border-radius: 20px;'>PyTorch</span>
    #                 <span style='background: #4ECDC4; color: white; padding: 0.5rem 1rem; border-radius: 20px;'>Streamlit</span>
    #                 <span style='background: #45B7D1; color: white; padding: 0.5rem 1rem; border-radius: 20px;'>InceptionV1</span>
    #                 <span style='background: #96CEB4; color: white; padding: 0.5rem 1rem; border-radius: 20px;'>Python</span>
    #             </div>
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.markdown("**🎭 CelebrityVision AI v2.0**")
    with footer_col2:
        st.markdown("**🤖 Powered by InceptionV1**")
    with footer_col3:
        st.markdown("**⭐ Star us on GitHub!**")

if __name__ == "__main__":
    main()