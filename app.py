import streamlit as st
import pandas as pd
import numpy as np
from integrated_farm_recommendations import (
    predict_yield, get_crop_recommendation, get_fertilizer_recommendation,
    get_pesticide_recommendation, get_water_management_recommendation,
    assess_weather_impact, assess_water_quality,
    irrigation_efficiency,
    crop_water_requirements,
    weather_impact,
    water_quality_parameters
)
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.set_page_config(page_title="Sustainable Farming Advisor", layout="wide")
    
    st.title("🌾 Sustainable Farming Recommendation System")
    
    tabs = st.tabs(["Farm Input", "Recommendations", "Analytics"])
    
    with tabs[0]:
        st.header("Farm Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_crop = st.selectbox(
                "Current Crop",
                ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Potato", "Soybean"]
            )
            
            st.subheader("Previous Crops")
            prev_crop1 = st.selectbox("Previous Crop (1 season ago)", 
                                    ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Potato", "Soybean"])
            prev_crop2 = st.selectbox("Previous Crop (2 seasons ago)", 
                                    ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Potato", "Soybean"])
            prev_crop3 = st.selectbox("Previous Crop (3 seasons ago)", 
                                    ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Potato", "Soybean"])
            
            soil_type = st.selectbox(
                "Soil Type",
                ["Loamy", "Clay", "Sandy", "Silty", "Peaty"]
            )
            
            season = st.selectbox(
                "Season",
                ["Kharif", "Rabi", "Zaid"]
            )
        
        with col2:
            organic_matter = st.slider("Organic Matter Content (%)", 0.0, 30.0, 2.0)
            soil_ph = st.slider("Soil pH", 0.0, 14.0, 7.0)
            
            fertilizer_type = st.multiselect(
                "Current Fertilizer Types",
                ["Urea", "NPK", "DAP", "MOP"]
            )
            
            fertilizer_category = st.selectbox(
                "Fertilizer Category",
                ["Chemical", "Organic", "Mixed"]
            )
            
            irrigation_type = st.selectbox(
                "Irrigation Type",
                ["Drip", "Sprinkler", "Flood", "Manual", "Rain-fed"]
            )
            
            farm_area = st.number_input("Farm Area (acres)", min_value=0.1, value=1.0)
        
        st.subheader("Weather and Water Parameters")
        col3, col4 = st.columns(2)
        
        with col3:
            temperature = st.slider("Current Temperature (°C)", -10.0, 50.0, 25.0)
            rainfall_level = st.selectbox("Rainfall Level", ["Low", "Moderate", "High"])
        
        with col4:
            water_ph = st.slider("Irrigation Water pH", 0.0, 14.0, 7.0)
            salinity_level = st.selectbox("Water Salinity Level", ["Low", "Moderate", "High"])
        
        st.subheader("Pesticide Information")
        current_pesticide = st.selectbox(
            "Current Pesticide Type",
            ["Synthetic Insecticides", "Chemical Fungicides", "Chemical Herbicides"]
        )
        
        pesticide_category = st.selectbox(
            "Pesticide Category",
            ["Chemical", "Organic", "Mixed"]
        )
    
    if st.button("Generate Recommendations"):
        with tabs[1]:
            st.header("Farm Recommendations")
            
            # 1. Crop Rotation
            with st.expander("🌱 Crop Rotation", expanded=True):
                next_crop = get_crop_recommendation(current_crop, 
                                                  [prev_crop1, prev_crop2, prev_crop3],
                                                  season, soil_type)
                unique_crops = len(set([current_crop, prev_crop1, prev_crop2, prev_crop3]))
                rotation_score = (unique_crops / 4) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Recommended next crop:** {next_crop}")
                    st.markdown(f"**Rotation Diversity Score:** {rotation_score:.1f}%")
                    st.progress(rotation_score/100)
                
                with col2:
                    # Crop rotation visualization
                    fig = go.Figure(data=[go.Pie(labels=[current_crop, prev_crop1, prev_crop2, prev_crop3],
                                                hole=.3,
                                                title="Crop History")])
                    st.plotly_chart(fig)
            
            # 2. Fertilizer Recommendations
            with st.expander("🌿 Fertilizer Management", expanded=True):
                fertilizer_recs = get_fertilizer_recommendation(
                    soil_type, current_crop, organic_matter, soil_ph,
                    ", ".join(fertilizer_type), fertilizer_category
                )
                
                # Display recommendations in a clean format
                for rec in fertilizer_recs:
                    if rec.startswith('\n'):
                        st.markdown("---")
                    st.markdown(rec)
                
                # Add soil health visualization
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=soil_ph,
                        title={'text': "Soil pH"},
                        gauge={'axis': {'range': [0, 14]},
                               'bar': {'color': "darkblue"},
                               'steps': [
                                   {'range': [0, 5.5], 'color': "red"},
                                   {'range': [5.5, 7.5], 'color': "green"},
                                   {'range': [7.5, 14], 'color': "red"}]}))
                    st.plotly_chart(fig)
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=organic_matter,
                        title={'text': "Organic Matter (%)"},
                        gauge={'axis': {'range': [0, 30]}}))
                    st.plotly_chart(fig)
            
            # 3. Pesticide Recommendations
            with st.expander("🐛 Pest Management", expanded=True):
                pesticide_recs = get_pesticide_recommendation(
                    current_pesticide, pesticide_category, current_crop, season
                )
                
                for rec in pesticide_recs:
                    if rec.startswith('\n'):
                        st.markdown("---")
                    st.markdown(rec)
            
            # 4. Water Management
            with st.expander("💧 Water Management", expanded=True):
                water_recs = get_water_management_recommendation(
                    current_crop, season, soil_type, irrigation_type, farm_area
                )
                
                for rec in water_recs:
                    if rec.startswith('\n'):
                        st.markdown("---")
                    st.markdown(rec)
                
                # Water efficiency visualization
                col1, col2 = st.columns(2)
                with col1:
                    efficiency = irrigation_efficiency.get(irrigation_type, 0.5) * 100
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=efficiency,
                        title={'text': "Irrigation Efficiency (%)"},
                        gauge={'axis': {'range': [0, 100]}}))
                    st.plotly_chart(fig)
                
                with col2:
                    if current_crop in crop_water_requirements and season in crop_water_requirements[current_crop]:
                        water_needed = crop_water_requirements[current_crop][season]
                        fig = go.Figure(go.Indicator(
                            mode="number+delta",
                            value=water_needed,
                            title={'text': "Water Requirement (mm/acre)"}))
                        st.plotly_chart(fig)
            
            # 5. Yield Prediction
            with st.expander("📊 Yield Prediction", expanded=True):
                yield_prediction = predict_yield(
                    current_crop, soil_type, season, organic_matter, soil_ph,
                    fertilizer_category, irrigation_type, farm_area,
                    temperature, rainfall_level, water_ph, salinity_level
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Estimated yield per acre:** {yield_prediction['per_acre']} tons")
                    st.markdown(f"**Total estimated yield:** {yield_prediction['total']} tons")
                    st.markdown(f"**Weather impact factor:** {yield_prediction['weather_impact']}")
                    st.markdown(f"**Water quality impact factor:** {yield_prediction['water_quality_impact']}")
                
                with col2:
                    # Impact factors visualization
                    impact_data = pd.DataFrame({
                        'Factor': ['Weather Impact', 'Water Quality'],
                        'Impact': [yield_prediction['weather_impact'], 
                                 yield_prediction['water_quality_impact']]
                    })
                    fig = px.bar(impact_data, x='Factor', y='Impact',
                                title="Yield Impact Factors")
                    st.plotly_chart(fig)
            
            # Additional Sustainable Practices
            with st.expander("🌍 Sustainable Practices", expanded=True):
                st.markdown("""
                1. Use crop residue as organic matter
                2. Implement mulching
                3. Consider companion planting
                4. Regular soil testing every 6 months
                5. Maintain field borders for beneficial insects
                """)

if __name__ == "__main__":
    main()