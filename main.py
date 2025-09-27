import streamlit as st
import json
import pandas as pd
import numpy as np
from guardLLM import get_scores
import matplotlib.pyplot as plt
import seaborn as sns

from multiAgent import *
st.set_page_config(layout="wide")

plt.style.use('dark_background')



with st.sidebar:
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

    st.subheader(":red[Model Details!]")
    st.text(" ")
    st.text(" ")
    llmModel=st.radio("The current LLM chosen",["Llama 3.2"])
    st.text(" ")
    llmValidator=st.radio("The current Validator model", ["Microsoft Phi-2"])
    st.text(" ")
    triggerNumber=st.slider("Maximum Trigger Threshold",1,10,5)


crews = crewAI(triggerNumber, "json")

def append():
    st.session_state["history"].append({"User": st.session_state["mess"]})



with st.columns([0.4,0.6])[1]:
    st.subheader(":red[LLM Manipulation DashBoard]")
st.text("")
st.text("")
st.text("")
if "history" not in st.session_state:
    st.session_state["history"]=[{"AI":"Hi, How can I help you!"}]



tabs=st.tabs([x.center(150,"\u2001") for x in ["LLM Manipulation","Prompt Analysis"]])

st.text(" ")

with tabs[0]:
    col_ = st.columns([0.4,0.4,0.2])
    with col_[0]:
        st.metric("Current LLM", "Llama 3.2")
    with col_[1]:
        st.metric("Validator LLM", "Microsoft Phi3")
    with col_[-1]:
        st.metric("Trigger Responses", f"{triggerNumber}")
    st.text("")
    st.text("")
    cols = st.columns([0.5,0.2, 0.5])
    with cols[0]:
        answer = None

        st.text("")
        col__=st.columns([0.8,0.2])
        with col__[0]:
            st.subheader(":blue[Chat with the LLM]")
        with col__[1]:
            if st.button("Clear Chat"):
                del st.session_state["history"]
                st.rerun()

        st.text("")
        st.text("")
        cont=st.container(height=400)
        with cont:
            for item in st.session_state["history"]:
                for x in item:
                    with st.chat_message(x):
                        st.write(item[x])
        message=st.chat_input("Please enter your query. I have a memory!",key="mess",on_submit=append)
        if message:
            answer = get_scores(message)
            response,retrigger,original = crews.response(message)
            st.session_state["history"].append({"AI":response})
            with cont.chat_message("AI"):
                st.write(response)
    with cols[-1]:

        if message:
            st.text("")
            st.subheader(":blue[Prompt/Response Analysis]")
            st.text(" ")
            st.text(" ")


            columnDivision=st.columns([0.4,0.6])
            with st.expander("View the Trigger Stats", expanded=False):
                st.write(f":green[The pipeline was triggered a total of {retrigger} times!]")
            st.text(" ")
            st.text(" ")
            with st.expander("View the original LLM Response",expanded=False):
                original=original.replace("\n","\u2028")
                st.write(f":green[{original}]")
            st.text(" ")
            st.text(" ")
            with st.expander("View the User Prompt Analysis",expanded=False):
                fig=plt.figure(figsize=(10,2))
                new={}
                for item in answer:
                    new[item]=answer[item].detach().numpy().tolist()[0][0]
                answer=new
                print(answer)
                plt.xticks(rotation=90)
                sns.barplot(x=list(answer.keys()),y=list(answer.values()))
                st.pyplot(fig)



with tabs[1]:
    st.text("")
    st.text("")
    st.text("")
    coll=st.columns([0.01,0.99])
    with coll[0]:
        pass

    with coll[1]:
        if "statsToxic.csv" not in os.listdir():
            with st.spinner("Performing the analysis:"):
                df=pd.read_csv("manipulativePrompts.csv")
                stats = []
                for item in df.iloc[:,0].tolist():

                    response,retrigger,original = crews.response(item)
                    stats.append([item,response,retrigger,original])
                    print(retrigger)
                pd.DataFrame(stats,columns=["Prompt","Final Response","Count","Original"]).to_csv("statsToxic.csv",index=False)


        dfToxic=pd.read_csv("statsToxic.csv")
        dfToxic = dfToxic.iloc[:, [0, -1, 1, 2]]

        st.subheader(":orange[Sample Prompts and Responses]")

        colCards=st.columns(3)
        with colCards[0]:
            cont1 = st.container(height=300,border=True)
            indices=dfToxic.iloc[0,:].index.tolist()
            indice=0
            for item in dfToxic.iloc[19,:]:
                cont1.write(f":green[{indices[indice]}]")
                if type(item)==str:
                    item=item.replace("\n","")
                cont1.write(f":blue[{item}]")
                st.text(" ")
                indice+=1
        with colCards[1]:
            cont2 = st.container(height=300,border=True)
            indices=dfToxic.iloc[0,:].index.tolist()
            indice=0
            for item in dfToxic.iloc[14,:]:
                cont2.write(f":green[{indices[indice]}]")
                if type(item)==str:
                    item=item.replace("\n","")
                cont2.write(f":blue[{item}]")
                st.text(" ")
                indice+=1
        with colCards[2]:
            cont3 = st.container(height=300,border=True)
            indices=dfToxic.iloc[14,:].index.tolist()
            indice=0
            for item in dfToxic.iloc[2,:]:
                cont3.write(f":green[{indices[indice]}]")
                if type(item)==str:
                    item=item.replace("\n","")
                cont3.write(f":blue[{item}]")
                st.text(" ")
                indice+=1






        stats=dfToxic.to_numpy().tolist()

        colsDivision=st.columns([0.5,0.1,0.4])


        st.text("")
        st.text("")

        with colsDivision[0]:
            st.subheader(":orange[Toxic Prompts Analysis:]")
            st.text(" ")
            st.text(" ")
            st.dataframe(dfToxic)
        with colsDivision[-1]:
            st.subheader(":orange[Retrigger Counts Graph:]")
            st.text(" ")
            st.text(" ")
            counts=[x[-1] for x in stats]
            fig=plt.figure(figsize=(10,4))
            counts.sort()
            plt.xlabel("Prompt Number")
            plt.ylabel("Trigger Count")
            sns.lineplot(x=list(range(len(counts))),y=counts)
            st.text("")
            st.text("")
            st.pyplot(fig)




