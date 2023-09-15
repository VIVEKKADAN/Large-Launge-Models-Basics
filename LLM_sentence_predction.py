import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def main():

    st.title("Sentence predction")

    # Create a text input box
    text = st.text_area("Enter your text here:", "")

    # Create a button to predict the next sentence
    if st.button("Predict Next Sentence"):
        # Predict the next sentence based on the input text
        predicted_sentence = predict_next_sentence(text)
        # Display the predicted sentence
        st.write("Predicted Next Sentence:")
        st.write(predicted_sentence)


def predict_next_sentence(input_text):
    # Encode the tokens of input text
    tokenized_input = tokenizer.encode(input_text, return_tensors="pt")

    # Model creation
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Set the model in evaluation mode to deactivate the DropOut modules
    model.eval()

    # Predict the next sentence
    with torch.no_grad():  # Disable gradient tracking
        output = model.generate(tokenized_input, max_length=50, num_return_sequences=1, pad_token_id=50256)

    # Decode the predicted sentence
    predicted_sentence = tokenizer.decode(output[0], skip_special_tokens=True)

    return predicted_sentence


if __name__ == "__main__":
    main()
