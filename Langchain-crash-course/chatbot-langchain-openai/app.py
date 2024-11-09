from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
import streamlit as st
output_parser=CommaSeparatedListOutputParser()
api='paste api key here'

chat_llm = ChatOpenAI(api_key=api, temperature=0.4)

role = st.text_input('Define role of chatbot')
question = st.text_input('Enter your Question')

chat_prompt = ChatPromptTemplate.from_messages([
    ('system', role),
    ('human', question)
])

chain = chat_prompt | chat_llm | output_parser

if st.button('submit'):
    output = chain.invoke({'text': question})
    st.write(output)