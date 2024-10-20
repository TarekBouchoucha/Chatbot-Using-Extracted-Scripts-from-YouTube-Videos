import os
from langchain.prompts import PromptTemplate
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_groq import ChatGroq

load_dotenv()

def answer_plant_question(question, chat_history):
    """
    Fetches relevant information from the ChromaDB plant database
    and uses a large language model to generate a comprehensive answer
    to the user's query.

    Args:
        question (str): The user's question about plants.

    Returns:
        str: The generated answer, combining retrieved information and LLM output.
    """

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chroma_client = chromadb.PersistentClient(path="My_database")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    collection = chroma_client.get_or_create_collection(name="Agriculture", embedding_function=sentence_transformer_ef)

    results = collection.query(
        query_texts=[question],
        n_results=3,
        include=['documents', 'distances', 'metadatas']
    )

    template = f"""
    You are a agriculture expert. 
    I will ask you a question and you will answer with all the necessary tricks and steps and anything related and ensure the information integrity.
    
    Below is the question / message:
    {question}

    Here is a list of similar informations you can use if needed:
    information 1 :
    {results['documents'][0][0]}
    information 2 :
    {results['documents'][0][1]}
    information 3 :
    {results['documents'][0][2]}

    if the question / message is not related to africulture answer naturally.
    if the message can have references and it's straight forward related to agriculture, write this at the end of the message without changing anything:
    
    \n** For more information check out these videos:\n
    * {results['metadatas'][0][1]["item_id"]}\n
    * {results['metadatas'][0][2]["item_id"]}\n
    * {results['metadatas'][0][1]["item_id"]}

    Please write the best response for the question:
    """

    prompt = PromptTemplate(
        input_variables=["message", "best_practice"],
        template=template
    )

    #chat without keeping history of the conversation
    chat_completion = client.chat.completions.create(
        messages =[
            {
                "role": "system",
                "content": "You are a agriculture expert."
            },
            {
                "role": "user",
                "content": template,
            }
        ],
        model="llama3-8b-8192",
    )
    #chat without keeping history of the conversation
    # chat_completion = client.chat.completions.create(
    #     messages = [
    #         {"role": m["role"], "content": m["content"]}
    #         for m in st.session_state.messages
    #         ] + [
    #         {
    #             "role": "user",
    #             "content": template,
    #         }
    #     ],
    #     model="llama3-8b-8192",
    # )
    
    answer = chat_completion.choices[0].message.content
    #print(answer)
    return answer

def weather(country="Bizerte"):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
    tools = load_tools(["openweathermap-api"], llm)
    agent_chain = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    result=agent_chain.run(f"What's the weather like in {country}?")
    return result

if __name__ == "__main__":

    st.title("Your agriculture assistant")

    st.chat_message("assistant").markdown("Hello there! "+ weather() + "How can I help you ?")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What you would like to know ?"):

        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        

        response=answer_plant_question(prompt, st.session_state.messages)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})