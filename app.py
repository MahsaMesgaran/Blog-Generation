import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


### Generating response from LLama2 model

def getLLamaResponse(input_text, no_words, blog_style):

    # LLama2 model
    llm = CTransformers(model = "model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type="llama",
                        config={"max_new_tokens":256,
                                "temperature":0.01})
    
    # Prompt template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        whithin {no_words} words.
        """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"],
                            template=template)
    
    # Generate the response from the LLama2 model which is from GGML
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)

    return response




### Using Streamlit to build a web app
 
st.set_page_config (page_title="Generage Blog",
                    page_icon="🧊",
                    layout="centered",
                    initial_sidebar_state="collapsed"
)
st.header("Generate Blog 🧊")


input_text = st.text_input("Enter the Blog Topic")

# Creating two coloumns for two additional fields
col1, col2 = st.columns([5,5])

with col1:
    no_words=st.text_input("No of words")
with col2:
    blog_style=st.selectbox("Writting the Blog for",
                            ("Researchers", "Data Scientists", "Common People"), index=0)

submit=st.button("Generate")

# Final Response
if submit:
    st.write(getLLamaResponse(input_text, no_words, blog_style))

