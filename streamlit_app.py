from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

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

csv_url = "https://raw.githubusercontent.com/Juleslassoeur/FarmingSys/main/df4.csv"

df = pd.read_csv(csv_url)
df['Produit'] = df['Produit'].replace(list_produits)
df['Maladie'] = df['Maladie'].replace(list_maladies)

df_display = pd.read_csv(csv_url)

array = df.values
X = array[:,0:5]
y = array[:,5:7]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
model = LinearRegression()
model.fit(X, y)

st.title("Modèle d'analyse des données agricoles")

st.write("Aperçu du data set : ")
st.write(df_display)


with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))
