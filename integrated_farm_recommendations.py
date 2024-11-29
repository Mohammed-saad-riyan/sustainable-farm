import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

df = pd.read_csv('sustainable_farming_dataset.csv')

crop_patterns = {
    'Rice': ['Wheat', 'Potato', 'Maize'],
    'Wheat': ['Rice', 'Soybean', 'Maize'],
    'Cotton': ['Wheat', 'Maize', 'Soybean'],
    'Maize': ['Wheat', 'Soybean', 'Rice'],
    'Sugarcane': ['Wheat', 'Soybean', 'Potato'],
    'Potato': ['Rice', 'Maize', 'Wheat'],
    'Soybean': ['Wheat', 'Rice', 'Maize']
}

fertilizer_mapping = {
    'Chemical': {
        'Urea': {'organic': 'Vermicompost', 'transition_time': '3-6 months'},
        'NPK': {'organic': 'Bio-fertilizer', 'transition_time': '4-6 months'},
        'DAP': {'organic': 'Compost', 'transition_time': '3-4 months'},
        'MOP': {'organic': 'Green Manure', 'transition_time': '2-3 months'}
    },
    'Mixed': {
        'Urea + Compost': {'organic': 'Compost', 'transition_time': '2-3 months'},
        'NPK + Bio-fertilizer': {'organic': 'Bio-fertilizer', 'transition_time': '1-2 months'},
        'DAP + Vermicompost': {'organic': 'Vermicompost', 'transition_time': '1-2 months'}
    }
}

pesticide_mapping = {
    'Chemical': {
        'Synthetic Insecticides': {'organic': 'Neem Oil', 'transition_time': '2-3 months'},
        'Chemical Fungicides': {'organic': 'Trichoderma', 'transition_time': '2-4 months'},
        'Chemical Herbicides': {'organic': 'Mulching + Manual Weeding', 'transition_time': '1-2 months'}
    },
    'Mixed': {
        'Limited Chemical + Neem': {'organic': 'Neem Oil', 'transition_time': '1-2 months'},
        'Integrated Pest Management': {'organic': 'Continue IPM', 'transition_time': 'Already sustainable'}
    }
}

crop_water_requirements = {
    'Rice': {'Kharif': 1200, 'Rabi': 1000, 'Zaid': 1400},
    'Wheat': {'Kharif': 450, 'Rabi': 400, 'Zaid': 500},
    'Cotton': {'Kharif': 700, 'Rabi': 650, 'Zaid': 800},
    'Maize': {'Kharif': 500, 'Rabi': 450, 'Zaid': 600},
    'Sugarcane': {'Kharif': 1500, 'Rabi': 1400, 'Zaid': 1700},
    'Potato': {'Kharif': 500, 'Rabi': 450, 'Zaid': 550},
    'Soybean': {'Kharif': 450, 'Rabi': 400, 'Zaid': 500}
}

irrigation_efficiency = {
    'Drip': 0.9,
    'Sprinkler': 0.75,
    'Flood': 0.6,
    'Manual': 0.5,
    'Rain-fed': 0.4
}

weather_impact = {
    'Temperature': {
        'Optimal': {'Rice': (25, 35), 'Wheat': (20, 25), 'Cotton': (21, 35), 
                   'Maize': (20, 30), 'Sugarcane': (25, 35), 'Potato': (15, 25), 
                   'Soybean': (20, 30)},
        'Impact': {'Low': 0.7, 'Optimal': 1.0, 'High': 0.8}
    },
    'Rainfall': {
        'Low': 0.8,
        'Moderate': 1.0,
        'High': 0.9
    }
}

water_quality_parameters = {
    'pH': {
        'Optimal': (6.5, 7.5),
        'Impact': {'Low': 0.8, 'Optimal': 1.0, 'High': 0.85}
    },
    'Salinity': {
        'Low': {'level': '<1000 ppm', 'impact': 1.0},
        'Moderate': {'level': '1000-2000 ppm', 'impact': 0.9},
        'High': {'level': '>2000 ppm', 'impact': 0.7}
    }
}

def get_crop_recommendation(current_crop, prev_crops, season, soil_type):
    if current_crop in crop_patterns:
        return np.random.choice(crop_patterns[current_crop])
    return "No specific recommendation available"

def get_fertilizer_recommendation(soil_type, current_crop, organic_matter, soil_ph, 
                                current_fertilizer, fertilizer_category):
    recommendations = []
    
    # Soil health recommendations
    if organic_matter < 2:
        recommendations.append("Low organic matter content:")
        recommendations.append("- Add compost or vermicompost")
        recommendations.append("- Consider green manuring")
    
    if soil_ph < 5.5:
        recommendations.append("\nAcidic soil conditions:")
        recommendations.append("- Add lime to increase pH")
        recommendations.append("- Use pH tolerant organic fertilizers")
    elif soil_ph > 7.5:
        recommendations.append("\nAlkaline soil conditions:")
        recommendations.append("- Add organic matter to balance pH")
        recommendations.append("- Consider sulfur application")
    
    # Fertilizer transition recommendations
    if fertilizer_category in fertilizer_mapping:
        if current_fertilizer in fertilizer_mapping[fertilizer_category]:
            organic_alt = fertilizer_mapping[fertilizer_category][current_fertilizer]
            recommendations.append(f"\nFertilizer transition plan:")
            recommendations.append(f"- Current: {current_fertilizer}")
            recommendations.append(f"- Recommended: {organic_alt['organic']}")
            recommendations.append(f"- Transition time: {organic_alt['transition_time']}")
    
    return recommendations

def get_pesticide_recommendation(current_pesticide, pesticide_category, crop, season):
    recommendations = []
    
    if pesticide_category in pesticide_mapping:
        if current_pesticide in pesticide_mapping[pesticide_category]:
            organic_alt = pesticide_mapping[pesticide_category][current_pesticide]
            recommendations.append(f"\nPesticide transition plan:")
            recommendations.append(f"- Current: {current_pesticide}")
            recommendations.append(f"- Recommended: {organic_alt['organic']}")
            recommendations.append(f"- Transition time: {organic_alt['transition_time']}")
    
    recommendations.append("\nIntegrated Pest Management (IPM) Practices:")
    recommendations.append("- Use pest monitoring and thresholds")
    recommendations.append("- Implement biological control methods")
    recommendations.append("- Practice crop rotation for pest management")
    recommendations.append("- Use companion planting for natural pest control")
    
    return recommendations

def predict_yield(crop, soil_type, season, organic_matter, soil_ph, 
                 fertilizer_category, irrigation_type, farm_area,
                 temperature, rainfall_level, water_ph, salinity_level):
    """Enhanced yield prediction including weather and water quality"""
    
    # Base yield ranges (tons per acre)
    base_yields = {
        'Rice': {'min': 2.5, 'max': 4.0},
        'Wheat': {'min': 1.8, 'max': 3.5},
        'Cotton': {'min': 0.8, 'max': 1.5},
        'Maize': {'min': 2.0, 'max': 3.8},
        'Sugarcane': {'min': 25.0, 'max': 35.0},
        'Potato': {'min': 8.0, 'max': 12.0},
        'Soybean': {'min': 1.2, 'max': 2.5}
    }
    
    if crop not in base_yields:
        return "Yield prediction not available for this crop"
    
    # Start with base yield
    base_yield = (base_yields[crop]['min'] + base_yields[crop]['max']) / 2
    
    # Soil type impact
    soil_factors = {
        'Loamy': 1.2,
        'Clay': 1.0,
        'Sandy': 0.8,
        'Silty': 1.1,
        'Peaty': 0.9
    }
    adjusted_yield = base_yield * soil_factors.get(soil_type, 1.0)
    
    # Season impact
    season_factors = {
        'Kharif': 1.0,
        'Rabi': 0.9,
        'Zaid': 0.8
    }
    adjusted_yield *= season_factors.get(season, 1.0)
    
    # Organic matter impact
    if organic_matter < 2:
        adjusted_yield *= 0.8
    elif organic_matter > 4:
        adjusted_yield *= 1.2
    
    # Soil pH impact
    if soil_ph < 5.5 or soil_ph > 7.5:
        adjusted_yield *= 0.9
    
    # Fertilizer impact
    fertilizer_factors = {
        'Chemical': 1.0,
        'Organic': 0.9,
        'Mixed': 0.95
    }
    adjusted_yield *= fertilizer_factors.get(fertilizer_category, 1.0)
    
    # Irrigation impact
    adjusted_yield *= irrigation_efficiency.get(irrigation_type, 0.7)
    
    # Weather impact
    weather_factor, _ = assess_weather_impact(crop, temperature, rainfall_level)
    adjusted_yield *= weather_factor
    
    # Water quality impact
    water_quality_factor, _ = assess_water_quality(water_ph, salinity_level)
    adjusted_yield *= water_quality_factor
    
    # Calculate total yield
    total_yield = adjusted_yield * farm_area
    
    return {
        'per_acre': round(adjusted_yield, 2),
        'total': round(total_yield, 2),
        'weather_impact': round(weather_factor, 2),
        'water_quality_impact': round(water_quality_factor, 2)
    }

def get_water_management_recommendation(crop, season, soil_type, irrigation_type, farm_area):
    """Generate water management recommendations"""
    
    recommendations = []
    
  
    if crop in crop_water_requirements and season in crop_water_requirements[crop]:
        water_needed = crop_water_requirements[crop][season]
        efficiency = irrigation_efficiency.get(irrigation_type, 0.7)
        

        total_water_needed = (water_needed * farm_area) / efficiency
        
        recommendations.append(f"\nWater Management Plan:")
        recommendations.append(f"- Base water requirement: {water_needed} mm/acre")
        recommendations.append(f"- Irrigation efficiency ({irrigation_type}): {efficiency*100}%")
        recommendations.append(f"- Total water needed: {total_water_needed:.2f} mm for {farm_area} acres")
        
        
        if irrigation_type == 'Drip':
            recommendations.append("\nRecommended irrigation schedule:")
            recommendations.append("- Daily light irrigation")
            recommendations.append(f"- Approximately {(water_needed/30):.1f} mm/day")
        elif irrigation_type == 'Sprinkler':
            recommendations.append("\nRecommended irrigation schedule:")
            recommendations.append("- Irrigate every 2-3 days")
            recommendations.append(f"- Approximately {(water_needed/15):.1f} mm per session")
        else:
            recommendations.append("\nRecommended irrigation schedule:")
            recommendations.append("- Irrigate every 5-7 days")
            recommendations.append(f"- Approximately {(water_needed/6):.1f} mm per session")
        
        
        recommendations.append("\nWater conservation measures:")
        recommendations.append("- Use mulching to reduce evaporation")
        recommendations.append("- Monitor soil moisture regularly")
        recommendations.append("- Irrigate during early morning or evening")
        
        if irrigation_type not in ['Drip', 'Sprinkler']:
            recommendations.append("\nSuggested improvements:")
            recommendations.append("- Consider upgrading to drip irrigation")
            recommendations.append("- Install soil moisture sensors")
            recommendations.append("- Implement rainfall harvesting")
    
    return recommendations

def assess_weather_impact(crop, temperature, rainfall_level):
    """Assess impact of weather conditions on yield"""
    impact_factors = []
    recommendations = []
    
   
    optimal_temp = weather_impact['Temperature']['Optimal'].get(crop, (20, 30))
    if temperature < optimal_temp[0]:
        temp_impact = weather_impact['Temperature']['Impact']['Low']
        recommendations.append(f"Temperature below optimal range for {crop}")
        recommendations.append("- Consider cold protection measures")
        recommendations.append("- Adjust planting time to warmer period")
    elif temperature > optimal_temp[1]:
        temp_impact = weather_impact['Temperature']['Impact']['High']
        recommendations.append(f"Temperature above optimal range for {crop}")
        recommendations.append("- Consider shade protection")
        recommendations.append("- Increase irrigation frequency")
    else:
        temp_impact = weather_impact['Temperature']['Impact']['Optimal']
        recommendations.append("Temperature in optimal range")
    
    
    rainfall_impact = weather_impact['Rainfall'].get(rainfall_level, 1.0)
    if rainfall_level == 'Low':
        recommendations.append("Low rainfall conditions:")
        recommendations.append("- Implement water conservation measures")
        recommendations.append("- Consider drought-resistant varieties")
    elif rainfall_level == 'High':
        recommendations.append("High rainfall conditions:")
        recommendations.append("- Ensure proper drainage")
        recommendations.append("- Monitor for disease pressure")
    
    return temp_impact * rainfall_impact, recommendations

def assess_water_quality(water_ph, salinity_level):
    """Assess impact of water quality on irrigation"""
    recommendations = []
    
    
    if water_ph < water_quality_parameters['pH']['Optimal'][0]:
        ph_impact = water_quality_parameters['pH']['Impact']['Low']
        recommendations.append("Low water pH:")
        recommendations.append("- Consider pH adjustment")
        recommendations.append("- Monitor soil pH regularly")
    elif water_ph > water_quality_parameters['pH']['Optimal'][1]:
        ph_impact = water_quality_parameters['pH']['Impact']['High']
        recommendations.append("High water pH:")
        recommendations.append("- Add acidifying agents to irrigation water")
        recommendations.append("- Monitor soil pH regularly")
    else:
        ph_impact = water_quality_parameters['pH']['Impact']['Optimal']
        recommendations.append("Water pH in optimal range")
    
    
    salinity_impact = water_quality_parameters['Salinity'][salinity_level]['impact']
    if salinity_level != 'Low':
        recommendations.append(f"\nWater salinity ({salinity_level}):")
        recommendations.append("- Monitor soil salinity")
        recommendations.append("- Consider salt-tolerant crops")
        if salinity_level == 'High':
            recommendations.append("- Implement leaching practices")
            recommendations.append("- Increase irrigation frequency")
    
    return ph_impact * salinity_impact, recommendations

def main():
    print("\n=== Integrated Sustainable Farming Recommendation System ===\n")
    
    
    current_crop = input("Enter current crop: ").capitalize()
    prev_crop1 = input("Enter previous crop (1 season ago): ").capitalize()
    prev_crop2 = input("Enter previous crop (2 seasons ago): ").capitalize()
    prev_crop3 = input("Enter previous crop (3 seasons ago): ").capitalize()
    
    print("\nSoil Types: Loamy, Clay, Sandy, Silty, Peaty")
    soil_type = input("Enter soil type: ").capitalize()
    
    print("\nSeasons: Kharif, Rabi, Zaid")
    season = input("Enter season: ").capitalize()
    
    
    organic_matter = float(input("\nEnter organic matter content (%): "))
    soil_ph = float(input("Enter soil pH: "))
    
    
    print("\nCurrent Fertilizer Types: Urea, NPK, DAP, MOP")
    current_fertilizer = input("Enter current fertilizer type: ").upper()
    
    print("\nFertilizer Categories: Chemical, Organic, Mixed")
    fertilizer_category = input("Enter fertilizer category: ").capitalize()
    
    
    print("\nCurrent Pesticide Types: Synthetic Insecticides, Chemical Fungicides, Chemical Herbicides")
    current_pesticide = input("Enter current pesticide type: ").title()
    
    print("\nPesticide Categories: Chemical, Organic, Mixed")
    pesticide_category = input("Enter pesticide category: ").capitalize()
    
    
    print("\nIrrigation Types: Drip, Sprinkler, Flood, Manual, Rain-fed")
    irrigation_type = input("Enter irrigation type: ").capitalize()
    
    farm_area = float(input("Enter farm area (acres): "))
    
    
    temperature = float(input("\nEnter current temperature (°C): "))
    print("\nRainfall Levels: Low, Moderate, High")
    rainfall_level = input("Enter rainfall level: ").capitalize()
    
    water_ph = float(input("\nEnter irrigation water pH: "))
    print("\nWater Salinity Levels: Low, Moderate, High")
    salinity_level = input("Enter water salinity level: ").capitalize()
    
    
    print("\n=== Comprehensive Farm Recommendations ===")
    
    
    next_crop = get_crop_recommendation(current_crop, 
                                      [prev_crop1, prev_crop2, prev_crop3],
                                      season, soil_type)
    print("\n1. Crop Rotation Recommendation:")
    print(f"- Recommended next crop: {next_crop}")
    
    
    unique_crops = len(set([current_crop, prev_crop1, prev_crop2, prev_crop3]))
    rotation_score = (unique_crops / 4) * 100
    print(f"- Rotation Diversity Score: {rotation_score:.1f}%")
    
   
    print("\n2. Fertilizer Recommendations:")
    fertilizer_recs = get_fertilizer_recommendation(
        soil_type, current_crop, organic_matter, soil_ph, 
        current_fertilizer, fertilizer_category
    )
    for rec in fertilizer_recs:
        print(rec)
    
  
    print("\n3. Pesticide Recommendations:")
    pesticide_recs = get_pesticide_recommendation(
        current_pesticide, pesticide_category, current_crop, season
    )
    for rec in pesticide_recs:
        print(rec)
    

    print("\n4. Yield Prediction:")
    yield_prediction = predict_yield(
        current_crop, soil_type, season, organic_matter, soil_ph,
        fertilizer_category, irrigation_type, farm_area,
        temperature, rainfall_level, water_ph, salinity_level
    )
    print(f"- Estimated yield per acre: {yield_prediction['per_acre']} tons")
    print(f"- Total estimated yield: {yield_prediction['total']} tons")
    print(f"- Weather impact factor: {yield_prediction['weather_impact']}")
    print(f"- Water quality impact factor: {yield_prediction['water_quality_impact']}")
    
    _, weather_recs = assess_weather_impact(current_crop, temperature, rainfall_level)
    print("\nWeather-based Recommendations:")
    for rec in weather_recs:
        print(rec)


    _, water_quality_recs = assess_water_quality(water_ph, salinity_level)
    print("\nWater Quality Recommendations:")
    for rec in water_quality_recs:
        print(rec)
    
    # 5. Water Management
    print("\n5. Water Management:")
    water_recs = get_water_management_recommendation(
        current_crop, season, soil_type, irrigation_type, farm_area
    )
    for rec in water_recs:
        print(rec)
    
    print("\nAdditional Sustainable Practices:")
    print("1. Use crop residue as organic matter")
    print("2. Implement mulching")
    print("3. Consider companion planting")
    print("4. Regular soil testing every 6 months")
    print("5. Maintain field borders for beneficial insects")

if __name__ == "__main__":
    try:
        while True:
            main()
            if input("\nWould you like another recommendation? (yes/no): ").lower() != 'yes':
                break
    except KeyboardInterrupt:
        print("\nThank you for using the recommendation system!")