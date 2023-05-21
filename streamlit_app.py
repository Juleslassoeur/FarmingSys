#librairies
import pandas as pd
import streamlit as st
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#données
list_cereales = {
    'Blé' : 1, 
    'Orge' : 2, 
    'Maïs' : 3, 
    'Tournesol' : 4, 
    'Colza' : 5, 
    'Soja' : 6
}

list_produits = {
    'Phyto A' : 1, 
    'Phyto B' : 2, 
    'Phyto C' : 3, 
    'Phyto D' : 4, 
    'Phyto E' : 5
}

list_maladies = {
    'A': 1, 
    'B': 2, 
    'C': 3, 
    'D': 4, 
    'E': 5,
    'F': 6
}


df = pd.read_csv("df4-2.csv")
df['Produit'] = df['Produit'].replace(list_produits)
df['Maladie'] = df['Maladie'].replace(list_maladies)
df_display = pd.read_csv("df4-2.csv")


#ML multi output 
array = df.values
X = array[:,0:5]
y = array[:,5:7]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
model = LinearRegression()
model.fit(X, y)


#web app
st.set_page_config(
    page_title="Farming System",
    page_icon="🌿",
)

st.title("Modèle d'analyse des données agricoles")

st.write("Aperçu du data set : ")
st.write(df_display)


#user input 
def get_user_input():
    vol_eau = st.sidebar.slider('Volume d\'eau', 100, 350, 200)
    temp = st.sidebar.slider('Température', 10, 35, 15)
    prod = st.sidebar.selectbox('Produit', options=list_produits)
    selected_prod = list_produits[prod]
    
    maladie = st.sidebar.selectbox('Maladie', options=list_maladies)
    selected_maladie = list_maladies[maladie]
    
    cereales = st.sidebar.selectbox('Céréale', options=list_cereales)
    selected_cereales = list_cereales[cereales]

    user_data = {
        'vol_eau' : vol_eau,
        'maladie' : selected_maladie,
        'temp' : temp,
        'prod' : selected_prod,
        'cereales' : selected_cereales
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()
prediction = model.predict(user_input)

def get_rendement_prediction(prediction):
    pred = round(prediction[0][0])
    
    if pred > 100:
        pred = 100
    return pred

rendement = get_rendement_prediction(prediction)


#résultats ML input user 
st.subheader('Résultats du modèle :')

value = user_input["prod"][0]
result = []

def get_all_data(user_inpupt):
    for i in list_produits.items():
        user_input2  = user_input.copy()
        user_input2['prod'].replace(user_input2["prod"][0] , i[1], inplace=True)
        pred = model.predict(user_input2)
        result.append(pred[0])

get_all_data(user_input)


result_list = [np.ndarray.tolist(element) for element in result] #convertir les arrays en listes
index = max(enumerate(result_list), key=lambda x: x[1][0])[0] #rendement max de tous les produits 

key = list(list_produits.keys())[list(list_produits.values()).index(value)] #index de la valeur max 


col1, col2, col3 = st.columns(3)
col1.metric("Rendement estimé", f"{rendement} %", round(result_list[index][0],2))
col2.metric("Quantité de produit", f"{round(prediction[0][1],2)} kg/ha", round(result_list[index][1],2) )
col3.metric("Produit", key, list(list_produits.keys())[index])

col1, col2 = st.columns(2)
col1.write('Donnée saisies par l\'utilisateur :')
col1.write(user_input)


col2.write('Output modèle :')
col2.write(prediction)

st.write('Analyse des résultats par produits :')

chart_data = pd.DataFrame(
    result_list,
    columns=['Rendement', 'Quantité'])

st.line_chart(chart_data)

print(result_list)




