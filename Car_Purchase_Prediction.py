# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 22:58:37 2023

@author: ahlem
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from PIL import Image

# Load the saved model
loaded_model = tf.keras.models.load_model("my_classifier.h5")

# Define a function for making a prediction on new data


def make_prediction(new_data):
    # Convert input data to Pandas DataFrame
    new_data = pd.DataFrame(new_data, columns=['Rating', 'Review Count', 'Price($)', 'Year', 'Brand', 'Model'])
    # Apply one-hot encoding to categorical variables
    new_data = pd.get_dummies(new_data, columns=['Brand', 'Model'], drop_first=True)
    # Scale the data using MinMaxScaler
    ms = MinMaxScaler()
    x = ms.fit_transform(new_data)
    # Check input shape and pad with zeros if necessary
    if x.shape[1] < loaded_model.input_shape[1]:
        diff = loaded_model.input_shape[1] - x.shape[1]
        zeros = np.zeros((x.shape[0], diff))
        x = np.hstack((x, zeros))
    # Make a prediction using the loaded model
    prediction = loaded_model.predict(x)
    # Return the predicted label
    return prediction[0]

# Define the Streamlit app
def app():
    
    
    
    html_temp="""
    <div style="background-color: lightblue; padding: 16px; ">
    <h2 style="color: black; text-align: center;">CarSalesForecast: Predicting Automotive Sales</h2>
    <div style="background-image: url("cars.jpeg"); padding: 16px; ">
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.write('')
    st.write('')
    # Ajouter un titre
    #st.title('CarSalesForecast: Predicting Automotive Sales)


    # Ajouter une description de la page
    st.markdown('**Welcome to our Car Sales Prediction website!**')
    st.markdown('**Please use the form below to enter information about the car you are considering to purchase, and our Deep Learning model will provide you with a recommendation on the deal\'s quality**')



    # Ajouter des instructions pour remplir le formulaire
    #st.markdown('Veuillez r:')

    # Set the title
    #st.title('Car Purchase Prediction Web App')
    
    # Ajouter un séparateur visuel pour une meilleure organisation
    st.markdown("---")
    
    #Ajouter des icônes pour les champs d'entrée de données
    col1, col2, col3 = st.columns(3)
    with col1:
       #chargement de l'image
       image = Image.open('Rating.png')
       resized_image = image.resize((60,60))
       
       #¶affiachge de l'image et de titre
       st.image(resized_image, width=60)
       #rating = st.text_input('Note de la voiture (sur 5)')
       rating = st.number_input('**Car Rating (out of 5)**', value=4.5, step=0.1)

    with col2:
       
       #chargement de l'image
       image = Image.open('Reviews.jpeg')
       resized_image = image.resize((60,60))
       
       #¶affiachge de l'image et de titre
       st.image(resized_image, width=60)
       #review_count = st.text_input('Nombre d\'avis sur la voiture')
       review_count = st.number_input('**Number of reviews on the car**', value=10)


    with col3:
       #st.image('price_icon.png')
       #chargement de l'image
       image = Image.open('Price.jpeg')
       resized_image = image.resize((60,60))
       #¶affiachge de l'image et de titre
       st.image(resized_image, width=60)
       #price = st.text_input('Prix de la voiture')
       price = st.number_input('**Car Price**', value=10000)

       
     
    # Ajouter une liste déroulante pour l'année
    year = st.selectbox('**Car Manufacturing Year**', range(2000, 2024))


    # Ajouter des suggestions automatiques pour la marque et le modèle
    brands = ['Acura','Audi', 'BMW', 'Buick' ,'Cadillac' ,'Chevrolet', 'Chrysler',
                  'Volkswagen', 'Dodge', 'Ford' ,'INFINITI', 'Honda' ,'GMC' ,'Kia' ,'Jeep',
                  'Jaguar' ,'Hyundai' ,'Mazda', 'Mercedes-Benz', 'Nissan', 'Porsche' ,'Toyota',
                  'Volvo' ,'FIAT' ,'Mitsubishi' ,'Ferrari' ,'RAM' ,'Subaru', 'Lexus']
    brand = st.selectbox('**Car Brand**', brands)
        
        
   # Créer un dictionnaire de modèles pour chaque marque
    brand_models = {
    'Acura': ['MDX','TLX', 'RDX', 'Integra, ILX','RLX','NSX'],
    'Audi': ['A1', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'Q2', 'Q3', 'Q5', 'Q7', 'Q8', 'e-tron', 'RS3', 'RS4', 'RS5', 'RS6', 'RS7', 'RS Q3', 'S1', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'SQ2', 'SQ5', 'SQ7', 'TT', 'TTS', 'TT RS'],
    'BMW': ['1 Series', '2 Series', '3 Series', '4 Series', '5 Series', '6 Series', '7 Series', '8 Series', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'Z4', 'M2', 'M3', 'M4', 'M5', 'M6', 'M8'],
    'Buick': ['Enclave', 'Encore', 'Encore GX', 'Envision', 'LaCrosse', 'Regal', 'Regal TourX'],
    'Cadillac': ['ATS', 'CT4', 'CT5', 'CT6', 'Escalade', 'XT4', 'XT5', 'XT6'],
    'Chevrolet': ['Camaro', 'Corvette', 'Equinox', 'Impala', 'Malibu', 'Silverado', 'Spark', 'Suburban', 'Tahoe', 'Trailblazer', 'Traverse', 'Volt'],
    'Chrysler': ['300', 'Pacifica', 'Voyager', 'Aspen', 'Crossfire', 'Sebring', 'PT Cruiser', 'Town & Country', 'Concorde', 'LHS', 'New Yorker', 'Cirrus', 'LeBaron'],
    'Volkswagen': ['Arteon', 'Atlas', 'Beetle', 'CC', 'e-Golf', 'Eos', 'Golf', 'GTI', 'Jetta', 'Passat', 'R32', 'Rabbit', 'Tiguan', 'Touareg'],
    'Dodge': ['Challenger', 'Charger', 'Durango', 'Grand Caravan', 'Journey', 'Neon', 'Nitro', 'Viper'],
    'Ford': ['Bronco', 'Edge', 'Escape', 'Expedition', 'Explorer', 'F-150', 'F-250', 'F-350', 'Fiesta', 'Flex', 'Focus', 'Fusion', 'GT', 'Mustang', 'Ranger', 'Taurus', 'Transit', 'Transit Connect'],
    'INFINITI': ['Q50', 'Q60', 'Q70', 'QX30', 'QX50', 'QX60', 'QX70', 'QX80'],
    'Honda': ['Accord', 'Civic', 'CR-V', 'Fit', 'HR-V', 'Insight', 'Odyssey', 'Passport', 'Pilot', 'Ridgeline'],
    'GMC': ['Acadia', 'Canyon', 'Sierra 1500', 'Sierra 2500HD', 'Sierra 3500HD', 'Terrain', 'Yukon', 'Yukon XL'],
    'Kia': ['Sorento', 'Soul', 'Telluride', 'Sportage', 'Optima', 'Stinger', 'Cadenza', 'Rio', 'Forte', 'K5', 'K900', 'Niro'],
    'Jeep': ['Cherokee', 'Compass', 'Gladiator', 'Grand Cherokee', 'Renegade', 'Wrangler'],
    'Jaguar': ['F-PACE', 'E-PACE', 'I-PACE', 'XF', 'XJ', 'F-TYPE'],
    'Hyundai': ['Accent', 'Elantra', 'Kona', 'Palisade', 'Santa Fe', 'Sonata', 'Tucson', 'Veloster', 'Venue'],
    'Mazda': ['CX-3', 'CX-30', 'CX-5', 'CX-9', 'MX-5 Miata', 'Mazda2', 'Mazda3', 'Mazda6'],
    'Mercedes-Benz': ['A-Class', 'B-Class', 'C-Class', 'CLA-Class', 'CLS-Class', 'E-Class', 'G-Class', 'GLA-Class', 'GLB-Class', 'GLC-Class', 'GLE-Class', 'GLS-Class', 'Metris', 'S-Class', 'SL-Class', 'SLS AMG', 'SLC-Class', 'Sprinter', 'X-Class'],
    'Nissan': ['Altima', 'Armada', 'Frontier', 'GT-R', 'Kicks', 'Leaf', 'Maxima', 'Murano', 'NV', 'NV200', 'Pathfinder', 'Rogue', 'Sentra', 'Titan', 'Versa'],
    'Porsche': ['911', '718 Cayman', '718 Boxster', 'Panamera', 'Macan', 'Cayenne', 'Taycan'],
    'Toyota': ['4Runner', '86', 'Avalon', 'C-HR', 'Camry', 'Corolla', 'GR Supra', 'Highlander', 'Land Cruiser', 'Mirai', 'Prius', 'Rav4', 'Sequoia', 'Sienna', 'Tacoma', 'Tundra', 'Venza', 'Yaris'],
    'Volvo': ['C30', 'C40 Recharge', 'C70', 'S40', 'S60', 'S60 Cross Country', 'S70', 'S80', 'S90', 'V40', 'V50', 'V60', 'V60 Cross Country', 'V70', 'V90', 'V90 Cross Country', 'XC40', 'XC60', 'XC70', 'XC90'],
    'FIAT': ['500', '500L', '500X', '124 Spider', 'Panda', 'Tipo', 'Doblo', 'Qubo', 'Freemont', 'Fullback', 'Punto', 'Fiorino', 'Strada', 'Talento'],
    'Mitsubishi': ['Eclipse Cross', 'Mirage', 'Outlander', 'Outlander PHEV', 'Outlander Sport', 'i-MiEV', 'Lancer', 'Lancer Evolution', 'RVR'] ,
    'Ferrari': ['360', '458 Italia', '458 Speciale', '488 GTB', '488 Pista', '575M Maranello', '599 GTB Fiorano', '612 Scaglietti', '812 Superfast', 'California', 'California T', 'F12 Berlinetta', 'F12tdf', 'F355', 'F430', 'F8 Tributo', 'FF', 'GTC4Lusso', 'LaFerrari', 'Monza SP', 'Portofino', 'SF90 Stradale'],
    'RAM': ['1500', '2500', '3500', '4500', '5500', 'ProMaster 1500', 'ProMaster 2500', 'ProMaster 3500', 'ProMaster City'],
    'Subaru': ['Ascent', 'BRZ', 'Crosstrek', 'Forester', 'Impreza', 'Legacy', 'Outback', 'WRX', 'STI'],
    'Lexus': ['ES', 'GS', 'GX', 'IS', 'LC', 'LS', 'LX', 'NX', 'RC', 'RX', 'UX']
    }

    # Obtenir les modèles de voiture correspondant à la marque sélectionnée
    models = brand_models[brand]

    # Afficher une liste déroulante pour sélectionner le modèle de la voiture
    model = st.selectbox('**Car Model**', models)
   

    # Convert the user input to a list
    new_data = [[rating, review_count, price, year, brand, model]]

    # Make a prediction on the new data
    if st.button('Predict'):
        prediction = make_prediction(new_data)

        # Interpret the prediction
        try:
            
            if prediction[0] == 0:
                
                result = "Fair Deal"
                result_color = "red"
            else:
                st.balloons()
                result = "Good Deal"
                result_color = "green"
                
        except:
            st.warning('Something Went Wrong please try again!')

        # Display the prediction result
        #st.success(result)
        
        st.markdown("---")
        st.write('## The prediction Result:')
        st.write(f"**According to the provided data, the prediction of the car is as follows {result}**", f"**for the given car with the Brand and the Moddel specified : {brand} {model}.**", 
             unsafe_allow_html=True, )
        st.write(f"**This prediction is generated based on the input features and the trained model used for car sales prediction**", unsafe_allow_html=True, )
        st.markdown(f"<p style='color:{result_color}; font-size: 32px;'>{result}</p>", unsafe_allow_html=True)


# Run the app
if __name__ == '__main__':
    app()
