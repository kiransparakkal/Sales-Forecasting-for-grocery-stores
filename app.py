import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from keras.models import load_model
from streamlit_option_menu import option_menu
import joblib


# Load the saved XGBoost model
model = joblib.load('model.pkl')

def label_encoder(data):
  columns_to_encode = ['Item_Fat_Content', 'Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']
  label_encoder = LabelEncoder()

  for column in columns_to_encode:
    data[column] = label_encoder.fit_transform(data[column]) + 1
  return data

def box_cox_transform(data):
    transformed_column, _ = stats.boxcox(data+ 1)  # Adding 1 to handle zero and negative values
    data = transformed_column
    return data

def min_max_scaler(data):
 col = ["Item_Fat_Content","Item_Type","Outlet_Size","Outlet_Type","Outlet_Location_Type","Item_Weight","Item_MRP","Outlet_Years"]
 for i in col:
    data[i] = (data[i]-data[i].min())/(data[i].max()-data[i].min())
 return data

def preprocess_data(data):
    # Substituting mean values in place of null values for 'Item Weight' column
    data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace=True)

    # Filling missing values in 'Outlet_Size' column with the mode
    data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0],inplace=True)

    zero_mask = (data['Item_Visibility'] == 0)
    
    # Handle zero values by replacing with the mean
    non_zero_values = data['Item_Visibility'][~zero_mask]
    mean_non_zero = non_zero_values.mean()
    data.loc[zero_mask,'Item_Visibility'] = mean_non_zero
    
    #Optimise 'Fat Content' column
    fat_content ={'low fat':'Low Fat',
              'LF':'Low Fat',
              'reg':'Regular'}
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(fat_content)

    # Changing oulet establishment year column with a new column 'Outlet Years' showing the total active years of the store
    data['Outlet_Years']=2023-data['Outlet_Establishment_Year']

    # Remove unwanted data
    data.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1,inplace=True)

    data = label_encoder(data)

    data['Item_Visibility'] = box_cox_transform(data['Item_Visibility'])

    data = min_max_scaler(data)

    return data

def post_process_data(user_input,predicted_sales):
   
   original_predicted_sales = np.power(predicted_sales, 3)
   #Add a new column with Initial inventory values
#    inventory_value = 1500  # Set your desired default value
#    user_input['Current_Inventory'] = inventory_value

   user_input['Predicted_Sales'] = original_predicted_sales
   user_input['Predicted_Sales'] = round(user_input['Predicted_Sales'])

   # Calculate the difference between columns
   user_input['Difference'] = user_input['Current_Inventory'] - user_input['Predicted_Sales']

   # Create a new column with 'negative' for negative differences
   user_input['Inventory_recommendation'] = user_input['Difference'].apply(lambda x: f' Order {(round((-x)))} items' if x < 0 else 'Sufficient stock present')

   return user_input





def main_content():
 # File upload
 uploaded_file = st.file_uploader("", type=["csv"])

 if uploaded_file is not None:
    # Read the uploaded CSV file
    input_df = pd.read_csv(uploaded_file)

    # Test dataset represents user_input to model, hence copying it to another dataframe for purpose of visualisation at the end
    required_input=input_df.drop('Current_Inventory',axis = 1)
    
    # Display uploaded data
    content_1 = """
         ### Uploaded Data:
           """
    st.markdown(rounded_content_box(content_1), unsafe_allow_html=True)

    st.write(input_df)
    
    processed_data = preprocess_data(required_input)
    # Make predictions using the model

    predictions = model.predict(processed_data)  # Make sure your model can handle the input format

    output_df = post_process_data(input_df,predictions)

    
    # Display predictions
    content_2 = """
         ### Predicted Sales:
           """
    st.markdown(rounded_content_box(content_2), unsafe_allow_html=True)
    st.write(output_df)
    
    # Download link for the output CSV file
    output_csv = output_df.to_csv(index=False)
    st.download_button("Download Output CSV", data=output_csv, file_name="output.csv")

background_image_path = "https://media2.giphy.com/media/83xGKHnlQa5yw/giphy.gif?cid=ecf05e47kwumawqt8ptconxzwnwahk2oz810k6snpbyhmc3b&ep=v1_gifs_related&rid=giphy.gif&ct=g"
# Function to set the content background with a background image
def set_content_background_with_image(image_path):
    """
    Sets the background of the content area with a background image.

    Parameters:
        image_url (str): URL or path of the background image.
    """
    style = f"""
        <style>
        
        .stApp {{
            background-image: url('{image_path}');
            background-size: cover;
            padding: 20px;  /* Add padding for better readability */
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


# Function to create a rounded-edge content box
def rounded_content_box(content):
    return f'''
        <div style="
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.1);
        ">
        {content}</div>
        
    '''



# def home_page():
selected = option_menu(None, ["Home","Predictor","Contact Us"], 
        icons=['house','bar-chart-fill','person-vcard'], menu_icon="cast", default_index=0,orientation="horizontal")
 
 # Home page content
with st.container():
    set_content_background_with_image(background_image_path)
    if selected == "Home":
#    st.title("Welcome to Big Mart Sales Predictor!👋")
   

      #st.sidebar.success("Select a demo above.")

     content = """
   ## Welcome to Big Mart Sales Predictor!👋
   Our web application is designed to empower grocery store owners and managers with a powerful tool for predicting sales and optimizing inventory management.
   With the ever-changing demands of the market, staying ahead is essential, and that's where our solution comes in.
   ### Why Use Our Application?
   Managing a grocery store involves juggling numerous variables, from product availability and demand fluctuations to inventory levels.
   Our Sales Forecasting Web Application takes the guesswork out of the equation. 
   By harnessing the power of advanced machine learning algorithms, we provide accurate predictions of future sales based on historical data and relevant features

   ### Key Features:

   * Easy Data Upload: Our user-friendly interface allows grocery store employees to effortlessly upload their product data.
   Whether it's item identifiers, weight, fat content, or any other attribute, our application streamlines the process.
   * Accurate Sales Predictions: Our advanced forecasting model takes into account a variety of factors, such as product visibility, type, price, and outlet characteristics. 
   This results in predictions that help you proactively manage inventory and plan for optimal stock levels.
   * Inventory Optimization: Making sure you have the right products on the shelves at the right time is crucial. 
   Our application's predictions enable you to adjust inventory levels according to predicted demand, reducing wastage and ensuring customer satisfaction.
   * User-Friendly Interface: We understand that not everyone is a data scientist. 
   That's why our interface is designed to be intuitive and accessible, allowing store employees to leverage the power of predictive analytics without needing extensive technical expertise.
   * Downloadable Results: After running predictions, you can easily download the results in CSV format. This makes it convenient to integrate the insights into your existing systems and processes.

   ### Sales Variable Decription
   * Item_Identifier ----- Unique product ID
   * Item_Weight ---- Weight of product
   * Item_Fat_Content ----- Whether the product is low fat or not
   * Item_Visibility ---- The % of the total display area of all products in a store allocated to the particular product
   * Item_Type ---- The category to which the product belongs
   * Item_MRP ----- Maximum Retail Price (list price) of the product
   * Outlet_Identifier ----- Unique store ID
   * Outlet_Establishment_Year ----- The year in which store store was established
   * Outlet_Size ----- The size of the store in terms of ground area covered
   * Outlet_Location_Type ---- The type of city in which the store is located
   * Outlet_Type ---- whether the outlet is just a grocery store or some sort of supermarket
   * Current_Inventory ---- Represents the current stock of the product in the store

    """
   # Render the rounded content box
     st.markdown(rounded_content_box(content), unsafe_allow_html=True)
   
   

    elif selected == "Predictor":
     # Streamlit app title
     content = """
    ## Sales Predictor
    
    This application includes a pre-trained forecasting model in its backend. This model generates predictions for sales values based on the provided data. Consequently, these predicted sales figures can be utilized to dynamically adjust and update inventory requirements.
    
    How It Works:

    * Data Upload: Simply upload your product data using our interface. Our application supports various attributes, making sure you can input the necessary information for accurate predictions.

    * Data Processing: The application processes the uploaded data, applying preprocessing techniques to ensure the input is ready for analysis.

    * Prediction: Our sophisticated forecasting model analyzes the data, taking historical sales and other attributes into account. This generates predictions for future sales.

    * Results Display: The predictions are presented in an easy-to-understand format. You can see which products are likely to experience higher demand in the upcoming periods.

    * Download and Action: Once you've reviewed the predictions, you can download the results as a CSV file. Use this information to fine-tune your inventory management strategies and make informed decisions.
    """
     st.markdown(rounded_content_box(content), unsafe_allow_html=True)
     main_content()

 # Other page content
    elif selected == "Contact Us":
     content = """
    ## Contact Us
    We value your interest in our Grocery Store Sales Forecasting Web Application. 
    If you have any questions, feedback, or inquiries, please don't hesitate to reach out to us. 
    Your input is important to us, and we're here to assist you.
    ### Contact Information:
    * Email: contact@grocerysalesforecast.com
    * Phone: +1 (123) 456-7890
    * Address: 123 Forecast Street, Cityville, State, Country
    
    """
     st.markdown(rounded_content_box(content), unsafe_allow_html=True)



