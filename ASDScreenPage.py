import streamlit as st
import pickle
import pandas as pd

def ChildPreprocessing15710(val):
  match val:
    case "Definitely Agree":
      return 1
    case "Slightly Agree":
      return 1
    case _:
      return 0


def ChildPreprocessing234689(val):
  match val:
    case "Definitely Disagree":
      return 1
    case "Slightly Disagree":
      return 1
    case _:
      return 0


def AdolescentPreprocessing15810(val):
  match val:
    case "Definitely Agree":
      return 1
    case "Slightly Agree":
      return 1
    case _:
      return 0


def AdolescentPreprocessing234679(val):
  match val:
    case "Definitely Disagree":
      return 1
    case "Slightly Disagree":
      return 1
    case _:
      return 0


def AdultPreprocessing17810(val):
  match val:
    case "Definitely Agree":
      return 1
    case "Slightly Agree":
      return 1
    case _:
      return 0


def AdultPreprocessing234569(val):
  match val:
    case "Definitely Disagree":
      return 1
    case "Slightly Disagree":
      return 1
    case _:
      return 0


def prediction(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, Age_Binned, Sex, Jaundice, Family_ASD):
  data = {
      "A1" : A1,
      "A2" : A2,
      "A3" : A3,
      "A4" : A4,
      "A5" : A5,
      "A6" : A6,
      "A7" : A7,
      "A8" : A8,
      "A9" : A9,
      "A10" : A10,
      "Age_Binned" : Age_Binned,
      "Sex" : Sex,
      "Jaundice" : Jaundice,
      "Family_ASD" : Family_ASD
  }

  predictdata = pd.DataFrame([data])

  predictdata["Sex"] = predictdata["Sex"].apply(lambda x: 'm' if x == "Male" else 'f')
  predictdata["Jaundice"] = predictdata["Jaundice"].str.lower()
  predictdata["Family_ASD"] = predictdata["Family_ASD"].str.lower()

  preprocesseddata = preprocess(predictdata)
  y_prob = bundle["ASDmodel"].predict(preprocesseddata)
  Class = (y_prob >= 0.5).astype(int)

  if Class[0] == 0:
    st.write("The person is not at risk of having (Autism Spectrum Disorder) ASD")
  else:
    st.write("The person is at risk of having (Autism Spectrum Disorder) ASD")


with open("ASDmodel.pkl", "rb") as f:
  bundle = pickle.load(f)


def apply_question_preprocessing(row, question_col):
    age_binned = row['Age_Binned']
    val = row[question_col]

    if age_binned == 'Child':
        if question_col in ['A1', 'A5', 'A7', 'A10']:
            return ChildPreprocessing15710(val)
        elif question_col in ['A2', 'A3', 'A4', 'A6', 'A8', 'A9']:
            return ChildPreprocessing234689(val)
    elif age_binned == 'Adolescent':
        if question_col in ['A1', 'A5', 'A8', 'A10']:
            return AdolescentPreprocessing15810(val)
        elif question_col in ['A2', 'A3', 'A4', 'A6', 'A7', 'A9']:
            return AdolescentPreprocessing234679(val)
    elif age_binned == 'Adult':
        if question_col in ['A1', 'A7', 'A8', 'A10']:
            return AdultPreprocessing17810(val)
        elif question_col in ['A2', 'A3', 'A4', 'A5', 'A6', 'A9']:
            return AdultPreprocessing234569(val)
    return val

def preprocess(data):
  for col in [f'A{i}' for i in range(1, 11)]:
    data[col] = data.apply(lambda row: apply_question_preprocessing(row, col), axis=1)
  data = bundle["encoder"].transform(data[bundle["columns"]])
  return data


st.title("AUTISM RISK SCREENING")
st.write("Choose a suitable answer from the options for the questions asked to predict whether a person is at the risk of having Autism Spectrum Disorder (ASD)")
st.write("It is suitable for people of age 4 upward with suspected autism who do not have a learning disability")

Sex = st.selectbox('If you are a parent/guardian answering for a child or adolescent, please select the child’s gender. If you are an adult participant, select your own gender', ['Male', 'Female'])
Jaundice = st.selectbox('Have you/the Child ever had Jauundice before?', ['Yes', 'No'])
Family_ASD = st.selectbox('Is there a family history of Autism Spectrum Disorder?', ['Yes', 'No'])
Age_Binned = st.selectbox('What age group are you in? Note: 4-11(Child), 12-17(Adolescent), 18+(Adult)', ['Child', 'Adolescent', 'Adult'])

#Ask ASD questions for each age group.
options = ['Definitely Disagree', 'Slightly Disagree', 'Slightly Agree', 'Definitely Agree']

A1 = A2 = A3 = A4 = A5 = A6 = A7 = A8 = A9 = A10 = None

if Age_Binned == 'Child':
  A1 = st.selectbox('She/he often notices small sounds when others do not ', options)
  A2 = st.selectbox('She/he usually concentrates more on the whole picture, rather than the small details', options)
  A3 = st.selectbox('In a social group, she/he can easily keep track of several different people’s conversations', options)
  A4 = st.selectbox('She/he finds it easy to go back and forth between different activities', options)
  A5 = st.selectbox('She/he doesn’t know how to keep a conversation going with his/her peers ', options)
  A6 = st.selectbox('She/he is good at social chit-chat', options)
  A7 = st.selectbox('When she/he is read a story, she/he finds it difficult to work out the character’s intentions or feelings', options)
  A8 = st.selectbox('When she/he was in preschool, she/he used to enjoy playing games involving pretending with other children', options)
  A9 = st.selectbox('She/he finds it easy to work out what someone is thinking or feeling just by looking at their face', options)
  A10 = st.selectbox('She/he finds it hard to make new friends', options)

elif Age_Binned == 'Adolescent':
  A1 = st.selectbox('She/he notices patterns in things all the time', options)
  A2 = st.selectbox('She/he usually concentrates more on the whole picture, rather than the small details', options)
  A3 = st.selectbox('In a social group, she/he can easily keep track of several different people’s conversations', options)
  A4 = st.selectbox('If there is an interruption, she/he can switch back to what she/he was doing very quickly', options)
  A5 = st.selectbox('She/he frequently finds that she/he doesn’t know how to keep a conversation going', options)
  A6 = st.selectbox('She/he is good at social chit-chat', options)
  A7 = st.selectbox('When she/he was younger, she/he used to enjoy playing games involving pretending with other children', options)
  A8 = st.selectbox('She/he finds it difficult to imagine what it would be like to be someone else', options)
  A9 = st.selectbox('She/he finds social situations easy', options)
  A10 = st.selectbox('She/he finds it hard to make new friends', options)

elif Age_Binned == 'Adult':
  A1 = st.selectbox('I often notice small sounds when others do not', options)
  A2 = st.selectbox('I usually concentrate more on the whole picture, rather than the small details', options)
  A3 = st.selectbox('I find it easy to do more than one thing at once', options)
  A4 = st.selectbox('If there is an interruption, I can switch back to what I was doing very quickly', options)
  A5 = st.selectbox('I find it easy to ‘read between the lines’ when someone is talking to me', options)
  A6 = st.selectbox('I know how to tell if someone listening to me is getting bored', options)
  A7 = st.selectbox('When I’m reading a story I find it difficult to work out the characters’ intentions', options)
  A8 = st.selectbox('I like to collect information about categories of things (e.g. types of car, types of bird, types of train, types of plant etc) ', options)
  A9 = st.selectbox('I find it easy to work out what someone is thinking or feeling just by looking at their face', options)
  A10 = st.selectbox('I find it difficult to work out people’s intentions', options)


if st.button('SCREEN AUTISM RISK'):
    if all(v is not None for v in [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]):
        prediction(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, Age_Binned, Sex, Jaundice, Family_ASD)
    else:
        st.error("Please answer all questions before screening")


