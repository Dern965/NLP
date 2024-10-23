import streamlit as st

# INTERFACE 1: Create Language Models
def interface_language_models():
    st.title("Create Language Models")
    
    # Load Corpus section
    st.subheader("Load corpus")
    corpus_input = st.text_input("Search", "")
    st.button("Browse")
    
    # Generate Bigrams and Trigrams buttons
    if st.button("Generate bigrams"):
        st.write("Bigrams generated!")
    
    if st.button("Generate trigrams"):
        st.write("Trigrams generated!")

# INTERFACE 2: Predictive Text
def interface_predictive_text():
    st.title("Predictive Text")
    
    # Load Language Model
    st.subheader("Load language model")
    model_input = st.text_input("Search", "")
    st.button("Browse")
    
    # Write a word or two
    st.subheader("Write a word (or two words to start a sentence)")
    word_input = st.text_input("Placeholder", "")
    
    # Next word and options
    if st.button("Next word"):
        st.write("Next word predicted!")
    st.selectbox("3 most probable words", ["Word1", "Word2", "Word3"])
    
    # Add word button
    st.button("Add word")
    
    # Generated text
    st.subheader("Generated text")
    st.text_area("Type here", "")

# INTERFACE 3: Text Generation
def interface_text_generation():
    st.title("Text Generation")
    
    # Load Corpus section
    st.subheader("Load corpus")
    corpus_input = st.text_input("Search", "")
    st.button("Browse")
    
    # Generate sentence button
    if st.button("Generate sentence"):
        st.write("Sentence generated!")
    
    # Generated text
    st.subheader("Generated text")
    st.text_area("Generated sentence", "")

# INTERFACE 4: Conditional Probability
def interface_conditional_probability():
    st.title("Conditional Probability")
    
    # Load Language Model section
    st.subheader("Load language model")
    model_input = st.text_input("Search", "")
    st.button("Browse")
    st.button("Add model")
    
    # Test sentence
    st.subheader("Test sentence")
    test_sentence = st.text_input("Placeholder", "")
    
    # Determine Joint Probability
    if st.button("Determine joint probability"):
        st.write("Joint probability calculated!")
    
    # Results
    st.subheader("Results")
    st.table({
        "Language model": ["Model 1", "Model 2", "Model n"],
        "Joint probability": [0.05, 0.04, 0.001]
    })

# MAIN PAGE NAVIGATION
st.sidebar.title("Interface Selection")
interface = st.sidebar.radio("Choose an Interface", 
                             ("Create Language Models", 
                              "Predictive Text", 
                              "Text Generation", 
                              "Conditional Probability"))

if interface == "Create Language Models":
    interface_language_models()
elif interface == "Predictive Text":
    interface_predictive_text()
elif interface == "Text Generation":
    interface_text_generation()
elif interface == "Conditional Probability":
    interface_conditional_probability()
