
#Let us use one LLM Model to poredict next word suggestion for the typed sentences using GPT2
import streamlit as st
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def main():
    # Set the title and header
    st.title("LLM Model")

    # Create a text input box
    text = st.text_area("Enter your text here:", "")

    # Create a button to process the text
    if st.button("Process"):
        # Process the input text (you can replace this with your own logic)
        output_text = process_text(text)
        # Display the output text
        st.write("Next Word:")
        st.write(output_text)
def process_text(text):
    # Encode the tokens of text inputs
    tokenindex = tokenizer.encode(text)
    # Convert indexed tokens in a PyTorch tensor format
    tokens_tensor = torch.tensor([tokenindex])
    # Model creation
    LLM_Model=GPT2LMHeadModel.from_pretrained('gpt2')
    # Set the model in evaluation mode to deactivate the DropOut modules
    LLM_Model.eval()


    # predicting all tokens
    with torch.no_grad():#to disable gradient tracking.
      output=LLM_Model(tokens_tensor)
      prediction=output[0]

    # Get the predicted next sub-words
    predicted_index = torch.argmax(prediction[0, -1, :]).item()
    predicted_text = tokenizer.decode(tokenindex + [predicted_index])

    return predicted_text

if __name__ == "__main__":
    main()
