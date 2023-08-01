import streamlit as st 
import requests
import numpy as np
import pandas as pd
import json

def main():
    st.title('Prêt à Dépenser - Scoring crédit')
    #response = requests.get('http://127.0.0.1:8080/api/users')
    response = requests.get('https://appscoring-64fa312b31ec.herokuapp.com/api/users')
    data = response.json()
    list_users = data['users']
    user_id = st.selectbox('Choisissez un ID de demande de prêt', list_users)
    if user_id == None:
        st.write("En attente d'un choix d'utilisateur")
    else:
        st.write('Vous avez choisi', user_id)
        #response2 = requests.get(f'http://127.0.0.1:8080/api/user/{user_id}')
        response2 = requests.get(f'https://appscoring-64fa312b31ec.herokuapp.com/api/user/{user_id}')
        user_data = response2.json()
        df = pd.DataFrame(user_data)
        st.dataframe(df.drop(columns=['proba']))
        st.write('')
        if st.button('Prédire'):
            st.write('Probabilité de remboursement :', f"{round(round(df['proba'].item(), 2)*100)}%")
            def result_loan(x):
                if x >= 0.6:
                    return 'Accordé'
                else:
                    return 'Non accordé'
            st.write('Réponse à la demande de prêt :', result_loan(df['proba'].item()))
            #response3 = requests.get('http://127.0.0.1:8080/api/shap_features')
            response3 = requests.get('https://appscoring-64fa312b31ec.herokuapp.com/api/shap_features')
            data2 = response3.json()
            df_shap = pd.DataFrame(data2)
            st.write('')
            st.write('Données déterminantes pour la prédiction :')
            for index in range(0,5):
                st.write(index + 1, df_shap[df_shap['SK_ID_CURR'] == user_id][str(index)].item())

        else:
            st.write('')


main()