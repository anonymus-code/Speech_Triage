import streamlit as st
from translate import CustomLLM  # Import the backend logic
llm = CustomLLM()

# Page title and description
st.title("Indic to English Translator")
st.write("""
    Translate text from various Indic languages (like Hindi) to English.
    Powered by the `ai4bharat/indictrans2-indic-en-1B` model.
""")

# Text input area
input_text = st.text_area(
    "Enter text in Hindi (or other Indic languages) to translate:",
    height=200,
)

# Translation button
if st.button("Translate"):
    if input_text.strip() == "":
        st.error("Please enter text to translate.")
    else:
        with st.spinner("Translating..."):

            # Perform the translation
            translation = llm._call(str(input_text))
            
            # Display the translation
            st.success("Translation:")
            st.write(input_text)
            st.write(translation)


