import pandas as pd 
import joblib
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='Employee Churn Prediction',
    page_icon='employee churn.jpeg'
                  )

st.markdown("<h1 style='text-align: center; color: blue;'>Employee Churn Prediction Application</h1>", unsafe_allow_html=True)





_, col2, _ = st.columns([2, 3, 2])

with col2:  
    img = Image.open("employee churn2.jpeg")
    st.image(img, width=300)

st.success('###### Our model is trained using 14,999 samples with the following parameters.')


st.info("###### Satisfaction Level: It is employee satisfaction point, which ranges from 0-1\n"
        "###### Last Evaluation: It is evaluated performance by the employer, which also ranges from 0-1\n"
        "###### Number of Projects: How many of projects assigned to an employee?\n"
        "###### Average Monthly Hours: How many hours in averega an employee worked in a month?\n"
        "###### Time Spent Company: time_spent_company means employee experience. The number of years spent by an employee in the company\n"
        "###### Work Accident: Whether an employee has had a work accident or not\n"
        "###### Promotion Last 5 years: Whether an employee has had a promotion in the last 5 years or not\n"
        "###### Departments: Employee's working department/division\n"
        "###### Salary: Salary level of the employee such as low, medium and high\n"
        "###### Left: Whether the employee has left the company or not")

st.success("###### Information about your employee")
 
    
col, col2 = st.columns([4, 4])
with col:
    st.markdown("###### Select Your Employee's Department")
    departments = st.selectbox("Departments",
            ('accounting', 'hr', 'IT', 'management', 'marketing', 'product_mng', 'RandD', 'sales', 'support', 'technical'  
            ))
    

with col2:
    st.markdown("###### Select Your Employee's Salary")
    salary = st.radio(
        "Salary",
        ('low', 'medium', 'high')
        )

        
col1, col2 = st.columns([4, 4])

with col1:
    st.markdown("#####")
    st.markdown("###### Has the employee had a work accident?")
    work_accident = st.radio(
        "Work Accident",
        ('Yes', 'No')
        )   

    if work_accident == "Yes":   
        work_accident = 1 
    elif work_accident == "No":     
        work_accident = 0


with col2:
    st.markdown("#####")
    st.markdown("###### Has the employee been promoted in the last 5 years?")   
    promotion_last_5years = st.radio(
        "Promotion Last 5 years",
        ('Yes', 'No')
        )   
if promotion_last_5years == "Yes":   
    promotion_last_5years = 1 
elif promotion_last_5years == "No":     
    promotion_last_5years = 0

    

    
satisfaction_level = st.sidebar.slider("Satisfaction level:", 0.0, 1.0, step=0.01, value=0.5)

last_evaluation = st.sidebar.slider("Last Evaluation Score:", 0.0, 1.0, step=0.01, value=0.5)

number_project = st.sidebar.slider("Number of Projects:",min_value=0, max_value=10)

average_montly_hours = st.sidebar.slider("Monthly Working Hours:",min_value=90, max_value=360)

time_spend_company = st.sidebar.slider("Years in the Company:",min_value=0, max_value=10)




my_dict = {
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_montly_hours': average_montly_hours,
    'time_spend_company': time_spend_company,
    'work_accident': work_accident,
    'promotion_last_5years': promotion_last_5years,  
    'departments': departments,
    'salary': salary
}

df=pd.DataFrame.from_dict([my_dict])


salary_map = {'high': '3', 'medium': '2' , 'low': '1'}
df['salary']=df['salary'].map(salary_map)
df['salary']=df['salary'].astype('int')


department_map = {'sales': '1', 'technical': '2' , 'support': '3','IT': '4', 'RandD': '5' , 'product_mng': '6','marketing': '7', 'accounting': '8' , 'hr': '9', 'management': '10'}
df['departments']=df['departments'].map(department_map)
df['departments']=df['departments'].astype('int')


model = joblib.load("model_rfc_churn.joblib")



st.info("Your Choices for the Employee")
my_dict



if st.button("Predict"):
    pred = model.predict(df)
    if int(pred[0]) == 1:
        st.error("Your employee has a high probability of churn")
        img = Image.open("churn.jpeg")
        st.image(img, width=300)
    else:
        st.success("Your employee loves the company")
        img = Image.open("not churn.png")
        st.image(img, width=300)           
