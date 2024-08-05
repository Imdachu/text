import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

def load_image(image_path):
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            return image
        else:
            print('Error: Could not open or find the image.')
    else:
        print(f"Error: Image file '{image_path}' does not exist.")
    return None

def classify_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    if des is None:
        return "Unknown"
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(des)
    texture_label = kmeans.predict(des)
    texture_classes = ['Sandy', 'Loamy', 'Clay']
    return np.random.choice(texture_classes)

def estimate_moisture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    if mean_intensity < 100:
        return "High Moisture"
    elif mean_intensity < 150:
        return "Moderate Moisture"
    else:
        return "Low Moisture"

def detect_nutrient_deficiency(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    yellow_percentage = (np.sum(mask > 0) / (image.shape[0] * image.shape[1])) * 100
    if yellow_percentage > 20:
        return "Possible Nitrogen Deficiency"
    else:
        return "No Significant Deficiency"

def analyze_ph(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:, :, 0])
    if mean_hue < 30:
        return "Acidic"
    elif mean_hue < 60:
        return "Neutral"
    else:
        return "Alkaline"

def analyze_color(image):
    mean_color = cv2.mean(image)
    return f"R: {mean_color[2]:.2f}, G: {mean_color[1]:.2f}, B: {mean_color[0]:.2f}"

def detect_plant_health(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
    green_percentage = (np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])) * 100
    if green_percentage > 50:
        return "Healthy Plants"
    else:
        return "Unhealthy Plants"

def detect_weeds(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    weed_mask = cv2.inRange(hsv, (25, 40, 40), (70, 255, 255))
    weed_percentage = (np.sum(weed_mask > 0) / (image.shape[0] * image.shape[1])) * 100
    if weed_percentage > 5:
        return "Weeds Detected"
    else:
        return "No Significant Weeds"

def estimate_organic_matter(image):
    # Simplified estimation based on color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brown_mask = cv2.inRange(hsv, (10, 60, 60), (30, 255, 255))
    brown_percentage = (np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])) * 100
    if brown_percentage > 10:
        return "High Organic Matter"
    else:
        return "Low Organic Matter"

def analyze_compaction(image):
    # Simplified analysis based on texture and intensity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    if edge_density > 0.1:
        return "Compacted Soil"
    else:
        return "Loose Soil"

def assess_erosion_risk(image):
    # Simplified risk assessment based on texture and color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    if edge_density > 0.2:
        return "High Risk of Erosion"
    else:
        return "Low Risk of Erosion"

def estimate_temperature_humidity(image):
    # Placeholder function for temperature and humidity estimation
    return "Temperature: 25°C, Humidity: 60%"

def detect_microbial_activity(image):
    # Placeholder function for microbial activity detection
    return "Moderate Microbial Activity"

def detect_heavy_metals(image):
    # Placeholder function for heavy metal detection
    return "No Heavy Metal Contamination"

def estimate_root_depth(image):
    # Placeholder function for root depth estimation
    return "Roots Depth: 20 cm"

def analyze_soil(image_path):
    image = load_image(image_path)
    if image is None:
        return
    
    texture = classify_texture(image)
    moisture = estimate_moisture(image)
    nutrient_deficiency = detect_nutrient_deficiency(image)
    soil_ph = analyze_ph(image)
    soil_color = analyze_color(image)
    plant_health = detect_plant_health(image)
    weeds = detect_weeds(image)
    organic_matter = estimate_organic_matter(image)
    compaction = analyze_compaction(image)
    erosion_risk = assess_erosion_risk(image)
    temperature_humidity = estimate_temperature_humidity(image)
    microbial_activity = detect_microbial_activity(image)
    heavy_metals = detect_heavy_metals(image)
    root_depth = estimate_root_depth(image)

    print(f"Soil Texture: {texture}")
    print(f"Moisture Level: {moisture}")
    print(f"Nutrient Deficiency: {nutrient_deficiency}")
    print(f"Soil pH: {soil_ph}")
    print(f"Soil Color: {soil_color}")
    print(f"Plant Health: {plant_health}")
    print(f"Weeds: {weeds}")
    print(f"Organic Matter Content: {organic_matter}")
    print(f"Soil Compaction: {compaction}")
    print(f"Erosion Risk: {erosion_risk}")
    print(f"Temperature and Humidity: {temperature_humidity}")
    print(f"Microbial Activity: {microbial_activity}")
    print(f"Heavy Metals: {heavy_metals}")
    print(f"Root Depth: {root_depth}")

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Texture: {texture}, Moisture: {moisture}, Nutrient Deficiency: {nutrient_deficiency}, pH: {soil_ph}, Color: {soil_color}, Plant Health: {plant_health}, Weeds: {weeds}\nOrganic Matter: {organic_matter}, Compaction: {compaction}, Erosion Risk: {erosion_risk}")
    plt.axis('off')
    plt.show()

# Analyze a sample soil image
image_path = 'C:\\project\\soil_new\\root_directory\\Yellow Soil\\27.jpg'

analyze_soil(image_path)
