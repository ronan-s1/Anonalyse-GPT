import streamlit as st
from main import *

st.title("Anonalyse GPT")
csv = st.file_uploader("Upload a CSV to analyse.", type=["csv"])

if csv:
    # getting fake and real dataframes
    real_df = pd.read_csv(csv, nrows=60)
    fake_df, column_mapping, table = create_fake_and_map(real_df)

    # display dummy columns and real columns
    st.write(table)
    query = st.text_area("")
    button = st.button("Submit")
    st.write("")

    # if user clicks submit
    if button:
        with st.spinner("Loading..."):
            generated_query = main(column_mapping, fake_df, query)
            query_result = eval(generated_query)

        st.code(generated_query)
        st.write(query_result)



    
