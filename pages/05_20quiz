import streamlit as st

# Initialize session state to track progress and answers
if 'question_index' not in st.session_state:
    st.session_state['question_index'] = 0
    st.session_state['answers'] = []
    st.session_state['correct_answer'] = "cat"  # Example correct answer
    st.session_state['game_over'] = False
    st.session_state['won'] = False

st.title("20 Questions Game 🧠")

# Display instructions
st.write("""
### Welcome to the 20 Questions Game!
You have 20 questions to guess the object in the image. Type your answer after each question.
""")

questions = [
    "Is it a living thing?",
    "Is it an animal?",
    "Is it a domestic animal?",
    "Is it smaller than a breadbox?",
    "Does it have fur?",
    "Does it make a sound?",
    "Can it fly?",
    "Is it a pet?",
    "Is it commonly kept in a house?",
    "Does it have a tail?",
    "Is it a mammal?",
    "Is it larger than a cat?",
    "Does it eat meat?",
    "Is it carnivorous?",
    "Is it nocturnal?",
    "Is it commonly found outdoors?",
    "Does it live in trees?",
    "Does it have sharp claws?",
    "Is it playful?",
    "Does it purr?",
]

# Display the current question
if st.session_state['question_index'] < 20:
    st.write(f"### Question {st.session_state['question_index'] + 1}:")
    st.write(questions[st.session_state['question_index']])
    
    # User input for their guess
    user_input = st.text_input("Your answer:", key=f"input_{st.session_state['question_index']}")
    
    # Proceed to the next question when the user submits their answer
    if st.button("Submit"):
        st.session_state['answers'].append(user_input)
        st.session_state['question_index'] += 1
        
        # Check if the user guessed the correct answer
        if user_input.lower() == st.session_state['correct_answer'].lower():
            st.session_state['won'] = True
            st.session_state['game_over'] = True
else:
    st.session_state['game_over'] = True

# Display game result
if st.session_state['game_over']:
    if st.session_state['won']:
        st.success("Congratulations! You guessed it right!")
    else:
        st.error(f"Game over! The correct answer was: {st.session_state['correct_answer']}")

# Button to reset the game
if st.button("Play Again"):
    st.session_state['question_index'] = 0
    st.session_state['answers'] = []
    st.session_state['game_over'] = False
    st.session_state['won'] = False
