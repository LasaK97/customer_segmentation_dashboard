import streamlit as st
st.set_page_config(page_title = 'Customer Segmentation',layout="wide", 
                   page_icon="icon/cus_page_icon.png") 

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
#import kaleido
import json

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.markdown("<h1 class = 'title'>Customer Segmentation via Cluster Analysis</h1>",unsafe_allow_html=True)
st.markdown("<hr style = 'color:red;'>",unsafe_allow_html=True) 
#st.sidebar.image("icon/cus_page_icon.png",width=100)

#Dataset path
dataset = "data/Amazon Sales FY2020-21.csv"

# To Improve speed and cache data
@st.experimental_memo 
def load_data(dataset):
	df = pd.read_csv(os.path.join(dataset))
	return df 

#Import dataframe

amazon_df = load_data(dataset)

column_names=['Order_Id', 'Order_Date', 'Status', 'Item_Id', 'SKU', 'Quantity_Ordered', 'Price', 'Value', 'Discount_Amount',
           'Total', 'Category', 'Payment_Method', 'By_St', 'Customer_Id', 'Year', 'Month',
           'Ref_Number', 'Name_Prefix', 'First_Name', 'Middle_Initial', 'Last_Name', 'Gender', 
           'Age', 'Full_Name', 'Email', 'Signed_Date', 'Phone_Number', 'Place_Name', 'County',
           'City', 'State', 'Zip_Code', 'Region', 'User_Name', 'Discount_Percent']
amazon_df.columns = column_names;
pd.set_option('display.max_columns', None)

st.sidebar.markdown("<hr>",unsafe_allow_html=True) 
st.sidebar.title("EDA")
st.sidebar.subheader("Dataset")

options1 = st.sidebar.selectbox('Basic Information',
                            ('---','Preview Dataframe','Show All Data Frame','Show All column Name',
                             'Dimensions of the Dataframe','Basic Statistics'))

if options1 == 'Preview Dataframe':
    
    button1, button2,  = st.columns([1,5])
    op = button1.selectbox('Select Head or Tail',('---','Head','Tail'))
    if op == 'Head':
        row = ('---',1,2,3,4,5,6,7,8,9,10)
        rows = button2.selectbox('Select No of Rows to be displayed',row)
        if rows != '---':
            st.write(amazon_df.head(rows)) # type: ignore
        else:
            st.warning("Please Select no of rows to be displayed !.")
    elif op == 'Tail' :
        row = ('---',1,2,3,4,5,6,7,8,9,10)
        rows = button2.selectbox('Select No of Rows to be displayed',row)
        if rows != '---':
            st.write(amazon_df.tail(rows)) # type: ignore
        else:
            st.warning("Please Select no of rows to be displayed !")
    elif op == '---':
        st.empty()
    else :
        st.warning("Please select Head or Tail !")
elif options1 == 'Show All Data Frame':
    st.dataframe(amazon_df)
elif options1 == 'Show All column Name':
    st.text("Columns:")
    st.write(list(amazon_df.columns))
elif options1 == 'Dimensions of the Dataframe':
    data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
    if data_dim == 'Rows':
        st.text("Showing Length of Rows")
        st.write(len(amazon_df))
    if data_dim == 'Columns':
        st.text("Showing Length of Columns")
        st.write(amazon_df.shape[1])
elif options1 == '---':
    st.empty()
else:
    st.write(amazon_df.describe())
    
#Making new dataframe with selected columns
new_df = amazon_df[['Order_Id', 'Quantity_Ordered', 'Price', 'Value', 'Discount_Amount', 'Total']]
pd.set_option('mode.chained_assignment', None)
new_df['New_Value']=new_df['Quantity_Ordered']*new_df["Price"]
new_df['New_Total'] = new_df['New_Value'] - new_df['Discount_Amount']
new_df.head(10)

#Updating Correct Total values and 
amazon_df['Total'] = new_df['Total']
amazon_df['Value'] =  new_df['New_Value']
amazon_df.head()

#Converting 'Order Date' in to datetime type
amazon_df["Order_Date"] = pd.to_datetime(amazon_df["Order_Date"])
print (amazon_df["Order_Date"].dtypes)

#coverting 'Signed_Date' to datetime type
amazon_df["Signed_Date"] = pd.to_datetime(amazon_df["Signed_Date"])
print (amazon_df["Signed_Date"].dtypes)

#Visualization
st.sidebar.subheader("Visualization")
variable_options = ['---','Correlation Matrix','No of Orders','Order Date','Product Category','Status','Gender',
                    'Region','State','City','Customers','Orders','Total']
variable = st.sidebar.selectbox("Select Visualize", variable_options, )

if variable == 'No of Orders':
    st.markdown("<h3 style='text-align = left;'>No of Orders </h3>",unsafe_allow_html=True)
    by = st.selectbox("Select By", ('---','Status','Category','Region & State'))
    c1, c2 = st.columns([4,1])
    if by == 'Status':
        
        status_counts = amazon_df['Status'].value_counts().sort_values(ascending=False)

        fig = go.Figure(go.Bar(
            x=status_counts.index,
            y=status_counts,
            marker=dict(
                color=px.colors.qualitative.Plotly,
            )))
        fig.update_layout({"title": 'No of Orders by Order Status',
                        "xaxis": {"title":"Order Status"},
                        "yaxis": {"title":" No of Orders"},
                        "showlegend": False}, 
                        title_x =0.285,
                        width=850,
                        height=500,
                        yaxis = dict(tickformat = "digits"),
                        template="plotly_dark",
                        title_font_family="Sitka Small",
                        title_font_size= 35,
                        )
        fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20,tickangle=35)
        fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
        
        with c1:
            st.write(fig)
        with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                    ##### Results
                        More orders are 
                        canceled than
                        completed.
                    
                    ''', unsafe_allow_html=True)    
        
    elif by == 'Category':
        category_counts = amazon_df['Category'].value_counts().sort_values(ascending=False)

        fig = go.Figure(go.Bar(
            x=category_counts.index,
            y=category_counts,
            marker=dict(
                color=px.colors.qualitative.Light24,
            )))
        fig.update_layout({"title": 'No of Orders by Category',
                        "xaxis": {"title":"Product Category"},
                        "yaxis": {"title":" No of Orders"},
                        "showlegend": False}, 
                        title_x =0.285,
                        width=850,
                        height=500,
                        yaxis = dict(tickformat = "digits"),
                        template="plotly_dark",
                        title_font_family="Sitka Small",
                        title_font_size= 35,
                        )
        fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20,tickangle=35)
        fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
        with c1:
            st.write(fig)
        with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                    ##### Results
                        Highest number 
                        of orders are
                        from the 'Mobiles 
                        and Tablets'
                        category.
                    
                    ''', unsafe_allow_html=True)   
    elif by == '---':
        st.empty()

    elif by == 'Region & State':
        
            fig = px.histogram(x = amazon_df['Region'].tolist(), color = amazon_df['State'].tolist())

            fig.update_layout({"title": 'No of Orders By Region and State',
                            "xaxis": {"title":"Region"},
                            "yaxis": {"title":"No of Orders"}}, 
                            title_x =0.2,
                            width=850,
                            height=500,
                            yaxis = dict(tickformat = "digits"),
                            legend_title="State",
                            template="plotly_dark",
                            title_font_family="Sitka Small",
                            title_font_size= 35,
                           
                            legend_title_font_color="white",
                            legend_title_font_size = 25,
                            )
            fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20)
            fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
            with c1:
                st.write(fig)
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                    ##### Results
                        Most of the 
                        orders are 
                        from southern
                        region. In
                        south most of
                        the orders are
                        from Texas.
                    
                    ''', unsafe_allow_html=True) 
        
    else:
        st.warning("Please select Chart Type")      
elif variable == "---":
     st.empty()   
     
elif variable == 'Status':
    st.markdown("<h3 style='text-align = left;'>Status </h3>",unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    op2 = c1.selectbox('Select Status',('---','Order Completed','Order Canceled'))
    data = amazon_df
    data['Gender'] = data['Gender'].replace(['M','F'],['Male', 'Female'])
    df_complete = data[data['Status'].str.contains('complete')]
    if op2 == 'Order Completed':
        df= df_complete.groupby('Gender')['Gender'].count()
        by = c2.selectbox('Select By',('---','Gender','Category','Region'))
        if by == '---':
            st.empty()
        
        elif by == 'Gender':
            st.markdown("<h3 style='text-align:'left'>Order Completed Customers by Gender</h3>",unsafe_allow_html=True)
            c1,c2 = st.columns([2,1])
            
            with c1:
                fig = go.Figure(data=[go.Pie(labels=df.index, values=df.values)])
                fig.update_layout({"title": 'Order Completed Customers by Gender',
                                "showlegend": False}, 
                                title_x =0.2,
                                width=500,
                                height=400,
                                yaxis = dict(tickformat = "digits"),
                                template="plotly_dark",
                                title_font_family="Sitka Small",
                                title_font_size= 15,
                                )
                fig.update_annotations(font_size=15)
                fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20,
                                marker = dict(line = dict(color = 'white', width = 3)))
                st.write(fig)
                
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                    ##### Results
                        49.4\u0025 females and 50.6\u0025 males are 
                        completed their orders.
                    
                    ''', unsafe_allow_html=True)
        elif by == 'Category':
                op3 = c3.selectbox('Chart Type',('---','Bar Chart','Treemap'))
                st.markdown("<h3 style='text-align:'left'>Order Completed Customers by Category</h3>",unsafe_allow_html=True)
                if op3 == '---':
                    st.empty()
                elif op3== 'Bar Chart':
                    c1,c2 = st.columns(2) 
                    with c1: 
                            category_counts = df_complete['Category'].value_counts().sort_values(ascending=False)  # type: ignore

                            fig = go.Figure(go.Bar(
                                x=category_counts.index,
                                y=category_counts,
                                marker=dict(
                                    color=px.colors.qualitative.Light24,
                                )))
                            fig.update_layout({"title": 'Completed Orders by Category',
                                            "xaxis": {"title":"Product Category"},
                                            "yaxis": {"title":" No of Orders"},
                                            "showlegend": False}, 
                                            title_x =0.285,
                                            width=550,
                                            height=450,
                                            yaxis = dict(tickformat = "digits"),
                                            template="plotly_dark",
                                            title_font_family="Sitka Small",
                                            title_font_size= 15,
                                            )
                            fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20,tickangle=35)
                            fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
                            st.write(fig)
                                
                    with c2:
                            with st.expander('', expanded=True):
                                st.markdown(f'''
                                ##### Results
                                    Most of the completed orders are from
                                    mobiles & Tablets Category.
                                
                                ''', unsafe_allow_html=True)             
                elif op3 == 'Treemap':
                    completed = df_complete.dropna()
                    fig = px.treemap(completed, 
                                    path=['Category'], template='plotly_dark')
                    fig.update_traces(textfont_color='yellow',textfont_size=16, selector=dict(type='treemap'))
                    fig.update_layout({"title": 'Completed Orders by Category',
                                    "showlegend": False}, 
                                    title_x =0.25,
                                    width=850,
                                    height=500,
                                    yaxis = dict(tickformat = "digits"),
                                    template="plotly_dark",
                                    title_font_family="Sitka Small",
                                    title_font_size= 25,
                                    )
                    
                    st.write(fig)
                else:
                    st.warning("Please Select Chart Type !") 
                
                
        elif by == 'Region':
            
            c1,c2 = st.columns(2)  
            with c1: 
                    category_counts = df_complete['Region'].value_counts().sort_values(ascending=False)  # type: ignore

                    fig = go.Figure(go.Bar(
                        x=category_counts.index,
                        y=category_counts,
                        marker=dict(
                        color=px.colors.qualitative.Light24,
                        )))
                    fig.update_layout({"title": 'Completed Orders by Region',
                                            "xaxis": {"title":"Region"},
                                            "yaxis": {"title":" No of Orders"},
                                            "showlegend": False}, 
                                            title_x =0.285,
                                            width=550,
                                            height=450,
                                            yaxis = dict(tickformat = "digits"),
                                            template="plotly_dark",
                                            title_font_family="Sitka Small",
                                            title_font_size= 15,
                                            )
                    fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20,tickangle=35)
                    fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
                    st.write(fig)
                                
                    with c2:
                        with st.expander('', expanded=True):
                            st.markdown(f'''
                            ##### Results
                                Most of the completed orders are from
                                Southern region.
                            
                            ''', unsafe_allow_html=True) 
        else:
            st.warning("Select proper variable")
    elif op2 == 'Order Canceled':
        df_cancel = data[data['Status'].str.contains('canceled')]
        df = df_cancel.groupby('Gender')['Gender'].count()
        by = c2.selectbox('Select By',('---','Gender','Category','Region'))
        if by == '---':
                st.empty()
        
        elif by == 'Gender':
                st.markdown("<h3 style='text-align:'left'>Order Canceled Customers by Gender</h3>",unsafe_allow_html=True)
                c1,c2 = st.columns([2,1])
                    
                with c1:
                    fig = go.Figure(data=[go.Pie(labels=df.index, values=df.values)])
                    fig.update_layout({"title": 'Order Canceled Customers by Gender',
                                    "showlegend": False}, 
                                    title_x =0.2,
                                    width=500,
                                    height=400,
                                    yaxis = dict(tickformat = "digits"),
                                    template="plotly_dark",
                                    title_font_family="Sitka Small",
                                    title_font_size= 10,
                                    )
                    fig.update_annotations(font_size=15)
                    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20,
                                    marker = dict(line = dict(color = 'white', width = 3)))
                    st.write(fig)
                        
                with c2:
                    with st.expander('', expanded=True):
                        st.markdown(f'''
                        ##### Results
                            49.6\u0025 females and 50.4\u0025 males are 
                            completed their orders.
                        
                        ''', unsafe_allow_html=True)
        elif by == 'Category':
                op3 = c3.selectbox('Chart Type',('---','Bar Chart','Treemap'))
                st.markdown("<h3 style='text-align:'left'>Order Canceled Customers by Category</h3>",unsafe_allow_html=True)
                if op3 == '---':
                    st.empty()
                elif op3== 'Bar Chart':
                    c1,c2 = st.columns(2)  
                    with c1: 
                            category_counts = df_cancel['Category'].value_counts().sort_values(ascending=False)  # type: ignore

                            fig = go.Figure(go.Bar(
                                x=category_counts.index,
                                y=category_counts,
                                marker=dict(
                                    color=px.colors.qualitative.Light24,
                                )))
                            fig.update_layout({"title": 'Canceled Orders by Category',
                                            "xaxis": {"title":"Product Category"},
                                            "yaxis": {"title":" No of Orders"},
                                            "showlegend": False}, 
                                            title_x =0.285,
                                            width=550,
                                            height=450,
                                            yaxis = dict(tickformat = "digits"),
                                            template="plotly_dark",
                                            title_font_family="Sitka Small",
                                            title_font_size= 15,
                                            )
                            fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20,tickangle=35)
                            fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
                            st.write(fig)
                                
                    with c2:
                            with st.expander('', expanded=True):
                                st.markdown(f'''
                                ##### Results
                                    Most of the canceled orders are from
                                    mobiles & Tablets Category.
                                </ul>
                                ''', unsafe_allow_html=True)             
                elif op3 == 'Treemap':
                    canceled = df_cancel.dropna()
                    fig = px.treemap(canceled, 
                                    path=['Category'], template='plotly_dark')
                    fig.update_traces(textfont_color='yellow',textfont_size=16, selector=dict(type='treemap'))
                    fig.update_layout({"title": 'Canceled Orders by Category',
                                    "showlegend": False}, 
                                    title_x =0.25,
                                    width=850,
                                    height=500,
                                    yaxis = dict(tickformat = "digits"),
                                    template="plotly_dark",
                                    title_font_family="Sitka Small",
                                    title_font_size= 25,
                                    )
                    
                    st.write(fig)
                else:
                    st.warning("Please Select Chart Type !") 
        elif by == 'Region':
            c1,c2 = st.columns(2)  
            with c1: 
                    category_counts = df_cancel['Region'].value_counts().sort_values(ascending=False)  # type: ignore

                    fig = go.Figure(go.Bar(
                        x=category_counts.index,
                        y=category_counts,
                        marker=dict(
                        color=px.colors.qualitative.Light24,
                        )))
                    fig.update_layout({"title": 'Canceled Orders by Region',
                                            "xaxis": {"title":"Region"},
                                            "yaxis": {"title":" No of Orders"},
                                            "showlegend": False}, 
                                            title_x =0.285,
                                            width=550,
                                            height=450,
                                            yaxis = dict(tickformat = "digits"),
                                            template="plotly_dark",
                                            title_font_family="Sitka Small",
                                            title_font_size= 15,
                                            )
                    fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20,tickangle=35)
                    fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
                    st.write(fig)
                                
                    with c2:
                        with st.expander('', expanded=True):
                            st.markdown(f'''
                            ##### Results
                                Most of the canceled orders are from
                                Southern region.
                            ''', unsafe_allow_html=True)                                    
        else:
                st.warning("Select proper variable")
    elif op2 == '---':
        st.empty()
    else:     
        st.warning("Select proper variable")
        
elif variable == 'Correlation Matrix':
    st.markdown("<h3 style='text-align:'left'>Correlation Matrix</h3>",unsafe_allow_html=True)
    
    c1,c2 = st.columns([2,1])
    corr_matrix = amazon_df.corr()
      
    corr = corr_matrix
    fig = ff.create_annotated_heatmap(
    z=corr.to_numpy().round(2),
    x=list(corr.index.values),
    y=list(corr.columns.values),       
    xgap=3, ygap=3,
    zmin=-1, zmax=1,
    colorscale='dense',
    colorbar_thickness=30,
    colorbar_ticklen=3
   
    )
    fig.update_layout(title_text='<b>Correlation Matrix<b>',
                  title_x=0.365,
                  titlefont={'size': 14},
                  width=600, height=600,
                  xaxis_showgrid=False,
                  xaxis={'side': 'bottom','color': 'white'},
                  yaxis_showgrid=False,
                  yaxis={'color': 'white'},
                  yaxis_autorange='reversed',                   
                  title_font_family="Sitka Small",
                  title_font_size= 15,
                  title_font_color= 'white',
    )  
    with c1:
        st.write(fig)
    
    with c2:
         with st.expander('', expanded=True):
                            st.markdown(f'''
                            ##### Results
                                Highest positive correlation
                                (0.95) exists between
                                'Total' and 'Value' variables.
                            ''', unsafe_allow_html=True)               
                            
elif variable == 'Customers':
    st.markdown("<h3 style='text-align:'left'>Customers</h3>",unsafe_allow_html=True)
    op = st.radio('Top 10 Customers By',('No of Orders','Total Expedinture'))
    if op == '':
        st.empty()
    elif op == 'No of Orders':
        top_cus_by_order = amazon_df.value_counts(['Customer_Id','Name_Prefix','First_Name','Last_Name',
                                        'Gender', 'Age','City','Region','User_Name'], sort=True).reset_index(name='count')
        st.write(top_cus_by_order.head(10))
    elif op == 'Total Expedinture':
        top_cus_by_total= amazon_df.groupby('Customer_Id').sum().sort_values(by=['Total'], ascending=False)
        st.write(top_cus_by_total.head(10))
    else:
        st.warning("Please Select !")
        
elif variable == 'Orders':
    st.markdown("<h3 style='text-align:'left'>Orders</h3>",unsafe_allow_html=True)
    op = st.radio('Orders per ',('Customer','Category'))
    if op == '':
        st.empty()
    elif op == 'Customer':
        c1,c2 = st.columns(2)  
        with c1: 
            number_of_orders = amazon_df.groupby('Customer_Id')['Order_Id'].nunique().sort_values(ascending=False)

            number_of_orders_df = pd.DataFrame(list(number_of_orders.items()), columns=['Customer ID', 'Number of Orders'])

            a = number_of_orders_df[number_of_orders_df['Number of Orders'] == 1].value_counts().sum()
            b = number_of_orders_df[number_of_orders_df['Number of Orders'] != 1].value_counts().sum()


            data = {'Order': ['One Order', 'More than One Order'], 'Customer_Counts': [a, b]}

            order_counts = pd.DataFrame.from_dict(data)

            fig = px.pie(order_counts, 
                        values = order_counts.Customer_Counts, 
                        names = order_counts.Order,
                        template = 'plotly_dark')

            fig.update_layout({"title": 'Orders Per Customer',
                            "showlegend": False}, 
                            title_x =0.25,
                            width=450,
                            height=450,
                            yaxis = dict(tickformat = "digits"),
                            template="plotly_dark",
                            title_font_family="Sitka Small",
                            title_font_size= 15,
                            )
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20,
                            marker = dict(line = dict(color = 'white', width = 3)))
            st.write(fig)   
        with c2:
            with st.expander('', expanded=True):
                st.markdown(f'''
                ##### Results
                    47.8\u0025 of customers have ordered 
                    more than one order.
            </ul>
            ''', unsafe_allow_html=True)  
    elif op == 'Category':
        c1,c2 = st.columns(2)  
        with c1: 
            number_of_prod = amazon_df.groupby('Customer_Id')['Category'].nunique().sort_values(ascending=False)

            number_of_prod_df = pd.DataFrame(list(number_of_prod.items()), columns=['Customer ID', 'Number of Products'])

            a = number_of_prod_df[number_of_prod_df['Number of Products'] == 1].value_counts().sum()
            b = number_of_prod_df[number_of_prod_df['Number of Products'] != 1].value_counts().sum()

            data = {'Order': ['One Category', 'More than One Category'], 'Customer_Counts': [a, b]}

            category_counts = pd.DataFrame.from_dict(data)

            fig = px.pie(category_counts, 
                        values = category_counts.Customer_Counts, 
                        names = category_counts.Order,
                        template = 'plotly_dark')
            fig.update_layout({"title": 'Orders Per Category',
                            "showlegend": False}, 
                            title_x =0.25,
                            width=450,
                            height=450,
                            yaxis = dict(tickformat = "digits"),
                            template="plotly_dark",
                            title_font_family="Sitka Small",
                            title_font_size= 15,
                            )
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20,
                            marker = dict(line = dict(color = 'white', width = 3)))
            st.write(fig)      
        with c2:
            with st.expander('', expanded=True):
                st.markdown(f'''
                ##### Results
                    24.5\u0025 of customers have ordered 
                    from more than one category.
            </ul>
            ''', unsafe_allow_html=True)  
    else:
        st.warning("Please Select !")
        
elif variable == 'Product Category':
    st.markdown("<h3 style='text-align:'left'>Product Category</h3>",unsafe_allow_html=True)
    op = st.selectbox('By',('---','Gender','Region'))
    c1,c2 = st.columns([4,1])
    if op == '---':
        st.empty()
    elif op == 'Gender':
           
        fig = px.bar(data_frame=amazon_df.groupby(by=["Category", "Gender"]).size().sort_values(ascending=False).reset_index(name="Counts"), 
            x="Gender", y="Counts", color="Category", barmode="group",
            title = "Count of each category with Gender"
        ) 

        fig.update_layout(
            font_family="Courier New",
            font_color="white",
            title_x =0.25,
            legend_title_font_color="white",
            legend_title_font_size = 15,
            autosize=False,
            width=850,
            height=500,
    
            legend=dict(
                bordercolor="white",
                borderwidth=1,
                
            ),
            yaxis = dict(tickformat = "digits"),
            template="plotly_dark",
            title_font_family="Sitka Small",
            title_font_size= 25,
        )
        fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
        with c1:
            st.write(fig)
        with c2:
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Results 
                    Both males and females
                    have ordered 'Mobiles
                    & Tablets' mostly.
                    ''', unsafe_allow_html=True)
    elif op == 'Region':
        
        fig = px.bar(data_frame=amazon_df.groupby(by=["Region", "Category"]).size().sort_values(ascending=False).reset_index(name="Counts"), 
            x="Region", y="Counts", color="Category", barmode="group",
            title = "Product Category By Region")
        

        fig.update_layout(
            font_family="Courier New",
            font_color="white",
            title_x =0.25,
            legend_title_font_color="white",
            legend_title_font_size = 15,
            autosize=False,
            width=850,
            height=500,
           
            legend=dict(
                bordercolor="white",
                borderwidth=1,
                
                
            ),
            yaxis = dict(tickformat = "digits"),
            template="plotly_dark",
            title_font_family="Sitka Small",
            title_font_size= 25,
        )
        fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20)
        fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
        with c1:
            st.write(fig)
        with c2:
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Results 
                    'Mobiles & Tablets' 
                     is the most
                     demanded category
                     in all the regions.   
                    ''', unsafe_allow_html=True)
    else:
        st.warning("Please Select !")
        
elif variable == 'Region':
    st.markdown("<h3 style='text-align:'left'>Region</h3>",unsafe_allow_html=True)
    c1,c2 = st.columns(2) 
    with c1: 
        fig = px.pie(amazon_df, values='Total', names='Region',color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout({"title": 'Region counts'}, 
                        title_x =0.2,
                        width=450,
                        height=450,
                        legend_title="Region",
                        legend_title_font_color="white",
                        legend_title_font_size = 15,
                        template="plotly_dark",
                        title_font_family="Sitka Small",
                        title_font_size= 25,
                            legend=dict(
                                        bordercolor="white",
                                        borderwidth=1,
                                        )
                    )
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20,
                        marker = dict(line = dict(color = 'white', width = 3)))
                        
        st.write(fig)                     
    with c2:
        with st.expander('', expanded=True):
            st.markdown(f'''
                ##### Results
                 Most of the  orders are from
                 Southern Region which is 38.4\u0025 
                 from total orders.
                ''', unsafe_allow_html=True)
            
elif variable == 'State':
    st.markdown("<h3 style='text-align:'left'>State</h3>",unsafe_allow_html=True) 
    states_counts = amazon_df['State'].value_counts().sort_values(ascending=False)

    op2 = st.selectbox('By',('---','Total Sales','No of Orders'))
    
    if op2 == '---':
        st.empty()
    elif op2 == 'No of Orders':
        
        fig = go.Figure(go.Bar(
            x=states_counts.index,
            y=states_counts,
            marker=dict(
                color='limegreen',
                line=dict(
                    color='white',  
                    width=3)
                    )
            ))
        fig.update_layout({"title": 'No of Orders by Order State',
                        "xaxis": {"title":"Product Category"},
                        "yaxis": {"title":" No of Orders"},
                        "showlegend": False}, 
                        title_x =0.3,
                        width=850,
                        height=600,
                        yaxis = dict(tickformat = "digits"),
                        template="plotly_dark",
                        title_font_family="Sitka Small",
                        title_font_size= 25,
                        )
        fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20,tickangle=35)
        fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
        st.write(fig)       
    elif op2 == 'Total Sales':
        usa_states = json.load(open('data/USA States/us-states.json','r'))
        total_for_state = amazon_df.groupby(['State']).agg({'Total':'sum'})
        fig = px.choropleth_mapbox(total_for_state, geojson=usa_states, locations=total_for_state.index, color=total_for_state['Total'],
                           color_continuous_scale="Viridis",
                           range_color=(0, total_for_state['Total'].max()),
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5
                          )


        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},width=850,
                   height=500,dragmode=False)
        
        st.write(fig)

    else:
        st.warning("Please Select chart type ! ")
elif variable == 'Order Date':
    st.markdown("<h3 style='text-align:'left'>Order Date</h3>",unsafe_allow_html=True) 
    op = st.selectbox('Select Chart Type',('---','Bar Chart','Line Chart'))
    by_month = pd.to_datetime(amazon_df['Order_Date']).dt.to_period('M').value_counts().sort_index()
    by_month.index = pd.PeriodIndex(by_month.index)
    df_month = by_month.rename_axis('Month').reset_index(name='Counts')
    c1, c2 = st.columns([4,1])
    if op == 'Line Chart':
        fig = go.Figure(data=go.Scatter(x=df_month['Month'].astype(dtype=str), 
                                y=df_month['Counts'],
                                marker_color='indianred', text="counts"))
        fig.update_layout({"title": 'Orders from Jan 2020 to Dec 2021',
                        "xaxis": {"title":"Months"},
                        "yaxis": {"title":"Total Orders"},
                        "showlegend": False}, 
                        title_x =0.25,
                        width=850,
                        height=500,
                        yaxis = dict(tickformat = "digits"),
                        template="plotly_dark",
                        title_font_family="Sitka Small",
                        title_font_size= 25,
                        )
        fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20)
        fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
        #fig.write_image("by-month.png",format="png", scale=1,engine='kaleido')
        
        with c1:
            st.write(fig)
        with c2:
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Results
                    Most of the orders
                    are ordered in 
                    December 2020.
                    ''', unsafe_allow_html=True)
             
    elif op == 'Bar Chart':
        fig = go.Figure(data=go.Bar(x=df_month['Month'].astype(dtype=str), 
                        y=df_month['Counts'],
                        marker=dict(
                            color='aqua'), 
                        text=df_month['Month'].dt.strftime('%b')))

        fig.update_layout({"title": 'Orders from Jan 2020 to Dec 2021',
                        "xaxis": {"title":"Months"},
                        "yaxis": {"title":"Total Orders"},
                        "showlegend": False}, 
                        title_x =0.25,
                        width=850,
                        height=500,
                        yaxis = dict(tickformat = "digits"),
                        template="plotly_dark",
                        title_font_family="Sitka Small",
                        title_font_size= 25,
                        )
        fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20)
        fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
        with c1:
            st.write(fig)
        with c2:
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Results
                    Most of the orders
                    are ordered in 
                    December 2020.
                    ''', unsafe_allow_html=True)
    elif op == '---':
        st.empty()
    else:
        st.warning("Please Select !")

elif variable == 'City':
    st.markdown("<h3 style='text-align:'left'>City</h3>",unsafe_allow_html=True) 
    op = st.selectbox('Top 10 Ordering',('Cities','Discounted Cities'))
    
    if op == 'Cities':
        top_cities = amazon_df.groupby('City').size().reset_index().rename(columns={0: 'Total'}).sort_values('Total', ascending=False).head(10)
        fig = px.pie(top_cities, values='Total', names='City', color_discrete_sequence=px.colors.sequential.RdBu, title='Top 10 ordering cities')
        fig.update_layout(
                        title_x =0.25,
                        width=800,
                        height=500,
                        legend_title = "City",
                        legend_title_font_color="white",
                        legend_title_font_size = 15,
                        template="plotly_dark",
                        title_font_family="Sitka Small",
                        title_font_size= 25,
                            legend=dict(
                                        bordercolor="white",
                                        borderwidth=1,
                                        )
                    )
        fig.update_traces(textposition='inside', textinfo='percent', textfont_size=20,
                        marker = dict(line = dict(color = 'white', width = 3)))
        st.write(fig)
    elif op == 'Discounted Cities':
        discounted_cities = amazon_df[amazon_df['Discount_Amount'] != 0]

        top_dis_cities = discounted_cities.groupby('City').size().reset_index().rename(columns={0: 'Total'}).sort_values('Total', ascending=False).head(10)
        fig = px.pie(top_dis_cities, values='Total', names='City', color_discrete_sequence=px.colors.sequential.BuGn_r, title='Top 10 ordering in discounted cities')
        fig.update_layout(
                        title_x =0.25,
                        width=900,
                        height=600,
                        legend_title = "City",
                        legend_title_font_color="white",
                        legend_title_font_size = 20,
                        template="plotly_dark",
                        title_font_family="Sitka Small",
                        title_font_size= 25,
                            legend=dict(
                                        bordercolor="white",
                                        borderwidth=1,
                                        )
                    )
        fig.update_traces(textposition='inside', textinfo='percent', textfont_size=20,
                        marker = dict(line = dict(color = 'white', width = 3)))
        st.write(fig)
    else:
        st.warning("Please Select !")  

elif variable == 'Gender':
    st.markdown("<h3 style='text-align:'left'>Gender</h3>",unsafe_allow_html=True) 
    op = st.selectbox('Select Gender By',('---','Category','Region'))
    data = amazon_df
    data['Gender'] = data['Gender'].replace(['M','F'],['Male', 'Female'])
    if op == '---':
         st.empty()
    elif op == 'Category':
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                     subplot_titles=("Male", "Female"))
        
        # Order completed

        df_males = data[data['Gender'].str.contains('Male')].groupby('Category')['Category'].count()
        fig.add_trace(go.Pie(labels=df_males.index, values=df_males .values, name='Male',hole=.55), 1, 1)

        # Order canceled
        df_females = data[data['Gender'].str.contains('Female')].groupby('Category')['Category'].count()
        fig.add_trace(go.Pie(labels=df_females.index, values=df_females, name='Female',hole=.55), 1, 2)

        # Update the layout of the subplot
        fig.update_layout({"title": 'Gender by Category',
                        "showlegend": True}, 
                        title_x =0.25,
                        width=800,
                        height=450,
                        legend_title = "Category",
                        legend_title_font_color="white",
                        legend_title_font_size = 15,
                        yaxis = dict(tickformat = "digits"),
                        template="plotly_dark",
                        title_font_family="Sitka Small",
                        title_font_size= 25,
                            legend=dict(
                                        bordercolor="white",
                                        borderwidth=1,
                        
                                        )
                        )
        fig.update_annotations(font_size=15)
        fig.update_traces(textposition='inside',hoverinfo='label+value', textinfo='percent', textfont_size=20,
                        marker = dict(line = dict(color = 'white', width = 2)))
        # Show the plot
        st.write(fig)
    
    elif op == 'Region':
        # Create the figure
        fig = px.histogram(
            x=amazon_df['Gender'].tolist(),
            color = amazon_df['Region'].tolist(),
            #orientation='h',
        )

        # Update the layout
        fig.update_layout({"title": 'Gender by Region',
                   "xaxis": {"title":"Gender"},
                   "yaxis": {"title":"No of Orders"}}, 
                   title_x =0.25,
                   width=850,
                   height=500,
                   yaxis = dict(tickformat = "digits"),
                   legend_title="Region",
                   template="plotly_dark",
                   title_font_family="Sitka Small",
                   title_font_size= 25,
                   legend=dict(
                          bordercolor="white",
                          borderwidth=1,
                               ),
                   legend_title_font_color="white",
                   legend_title_font_size = 15,
                   )
        fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20)
        fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
        # Show the figure
        st.write(fig)
    else:
       st.warning("Please Select a variable !")
elif variable == 'Total':
    st.markdown("<h3 style='text-align:'left'>Total</h3>",unsafe_allow_html=True) 
    c1,c2 = st.columns(2)
    op = c1.selectbox('Select By',('---','Discount','Category','Region & State','Gender'))
    
    if op == '---':
        st.empty()
    elif op == 'Discount':
        op1 = c2.selectbox('Select By',('---','Category','Region','Category & Region'))
        if op1 == '---':
            fig = px.scatter(amazon_df, x='Discount_Amount', y='Total')
            fig.update_layout({"title": 'Price vs Discount',
                        "xaxis": {"title":"Discount"},
                        "yaxis": {"title":"Total"}}, 
                        title_x =0.35,
                        width=850,
                        height=500,
                        yaxis = dict(tickformat = "digits"),
                        xaxis = dict(tickformat = "digits"),
                        title_font_family="Sitka Small",
                        title_font_size= 25,
                        )
            fig.update_traces(marker=dict(
                                                color='#4ad295'
                                                )
                                )
            fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20)
            fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
            c1.write(fig)
        elif op1 =='Category':
                fig = px.scatter(amazon_df, x='Discount_Amount', y='Total',color="Category")
                fig.update_layout({"title": 'Price vs Discount vs Category',
                            "xaxis": {"title":"Discount"},
                            "yaxis": {"title":"Total"}}, 
                            title_x =0.25,
                            width=850,
                            height=500,
                            yaxis = dict(tickformat = "digits"),
                            xaxis = dict(tickformat = "digits"),
                            title_font_family="Sitka Small",
                            title_font_size= 25,
                            )
                fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20)
                fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
                c1.write(fig)
        elif op1 == 'Region':
                fig = px.scatter(amazon_df, x='Discount_Amount', y='Total',color="Region")
                fig.update_layout({"title": 'Price vs Discount vs Region',
                            "xaxis": {"title":"Discount"},
                            "yaxis": {"title":"Total"}}, 
                            title_x =0.25,
                            width=850,
                            height=500,
                            yaxis = dict(tickformat = "digits"),
                            xaxis = dict(tickformat = "digits"),
                            title_font_family="Sitka Small",
                            title_font_size= 25,
                            )
                fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20)
                fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
                c1.write(fig)
        elif op1 == 'Category & Region':
                fig = px.scatter(amazon_df, x='Discount_Amount', y='Total',color='Category',symbol='Region')
                fig.update_layout({"title": 'Price vs Discount vs Category vs Region',
                            "xaxis": {"title":"Discount"},
                            "yaxis": {"title":"Total"}}, 
                            title_x =0.2,
                            width=850,
                            height=500,
                            yaxis = dict(tickformat = "digits"),
                            xaxis = dict(tickformat = "digits"),
                            title_font_family="Sitka Small",
                            title_font_size= 25,
                            
                            )
                fig.update_xaxes(title_font_family="Sitka Small", title_font_size= 20)
                fig.update_yaxes(title_font_family="Sitka Small", title_font_size= 20)
                c1.write(fig)
        else:
            st.warning("Please Select !")    
        
    elif op == 'Region & State':
        df_places = amazon_df.groupby(['Region', 'State']).sum().reset_index()
        df_places.sort_values(by='Total', ascending=False, inplace=True)
        df_places.head() 
        fig = px.sunburst(data_frame=df_places, path=['Region', 'State'], values='Total', title = 'Total Amount By Region and State')

        fig.update_layout(
                        title_x =0.25,
                        width=850,
                        height=500,
                        yaxis = dict(tickformat = "digits"),
                        template='plotly_dark',
                        title_font_family="Sitka Small",
                        title_font_size= 25,
                        )


        st.write(fig)
    else :
         st.warning("Select proper variable")   
else:
    st.warning("Select proper variable")

st.sidebar.markdown('---',unsafe_allow_html=True)    

#Clustering
st.sidebar.title('Clustering')

option2 = st.sidebar.selectbox('Select Clustering technique',('---','K-Mean','Agglomerative','Gaussian Mixture','K-Mode'))

if option2 == '---':
    st.empty()
    #KMeas
elif option2 == 'K-Mean':
    st.markdown("<h3 style='text-align:'left'>K Mean Clustering</h3>",unsafe_allow_html=True)
    op = st.selectbox('Select Data',('---','Order Completed Data','Order Canceled Data'))
    if op == '---':
        st.empty()
    elif op == 'Order Completed Data':
        st.markdown("<h4 style='text-align:'left'>Order Completed Data",unsafe_allow_html=True) 
        c1, c2 = st.columns([2,1])
        with c1:
            st.image("Clustering/complete_df/k_mean_K.png")
        with c2:
            
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Results
                    Optimal number of K is 6.
                    ''', unsafe_allow_html=True)
                
        st.markdown("<hr>",unsafe_allow_html=True) 
        op1 = st.selectbox('Select Graph',('---','Bar chart','3D Scatter plot','Snake Plot'))
        
        if op1 == '---':
            st.empty()
        elif op1 == 'Bar chart':
            st.image("Clustering/complete_df/k_mean_bar.png")
        elif op1 == '3D Scatter plot':
            st.image("Clustering/complete_df/k_3d.png")
        elif op1 == 'Snake Plot': 
           
            st.image("Clustering/complete_df/k_mean_snake.png")
        else:
            st.warning("Please select a chart !")
        
        st.markdown("<hr>",unsafe_allow_html=True) 
        op1 = st.selectbox('Visualize predicted clusters By',('---','Region','Category','Gender'))
        
        if op1 == '---':
            st.empty()
        elif op1 == 'Region':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/complete_df/kmean Region.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In all the clusters
                        most of the customers are 
                        from southern region.
                        ''', unsafe_allow_html=True) 
        elif op1 == 'Category':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/complete_df/kmean category.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In cluster 1 & 2 most 
                        demanded category is 
                        'Mens fashion' but in
                        other clusters it is
                        'Mobiles & Tablets'
                        ''', unsafe_allow_html=True)
        elif op1 == 'Gender':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/complete_df/kmean gender.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In cluster 1, 2 & 3 most
                        of the orders are ordered by 
                        Meles. 
                        ''', unsafe_allow_html=True)
        else:
            st.warning("Please select a chart !")
            
        
    elif op == 'Order Canceled Data':
        st.markdown("<h4 style='text-align:'left'>Order Canceled Data",unsafe_allow_html=True) 
        c1, c2 = st.columns([2,1])
        with c1:
            st.image("Clustering/cancel_df/k_mean_k.png")
        with c2:
            
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Results
                    Optimal number of K is 6.
                    ''', unsafe_allow_html=True)
                
        st.markdown("<hr>",unsafe_allow_html=True) 
        op1 = st.selectbox('Select Graph',('---','Bar chart','3D Scatter plot','Snake Plot'))
        
        if op1 == '---':
            st.empty()
        elif op1 == 'Bar chart':
            st.image("Clustering/cancel_df/k_mean_bar.png")
        elif op1 == '3D Scatter plot':
            st.image("Clustering/cancel_df/k_3d.png")
        elif op1 == 'Snake Plot': 

            st.image("Clustering/cancel_df/k_mean_snake.png")
                    
        else:
            st.warning("Please select a chart !")
        
        st.markdown("<hr>",unsafe_allow_html=True) 
        op1 = st.selectbox('Visualize predicted clusters By',('---','Region','Category','Gender'))
        
        if op1 == '---':
            st.empty()
        elif op1 == 'Region':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/cancel_df/kmean Region.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In all the clusters
                        most of the customers are 
                        from southern region.
                        ''', unsafe_allow_html=True) 
        elif op1 == 'Category':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/cancel_df/kmean category.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In cluster 1, 2, 3 & 5 most 
                        canceled category is 'Mobiles
                        & Tablets' but in cluster 0 & 4
                        it is 'Others'.
                        ''', unsafe_allow_html=True)
        elif op1 == 'Gender':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/cancel_df/kmean gender.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        Both males and females have
                        canceled nearly equivalent number 
                        of orders.
                        ''', unsafe_allow_html=True)
        else:
            st.warning("Please select a chart !")
    else:
         st.warning("Please Select Data !")
    
    #KMode
elif option2 == 'K-Mode':
    st.markdown("<h3 style='text-align:'left'>K Mode Clustering</h3>",unsafe_allow_html=True)
    op = st.selectbox('Select Data',('---','Order Completed Data','Order Canceled Data'))
    if op == '---':
        st.empty()
    elif op == 'Order Completed Data':
        st.markdown("<h4 style='text-align:'left'>Order Completed Data",unsafe_allow_html=True) 
        c1, c2 = st.columns(2)
        with c1:
            st.image("Clustering/complete_df/kmode_elbow.png")
        with c2:
            
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Results
                    Optimal number of K is 10.
                    ''', unsafe_allow_html=True)
                
        st.markdown("<hr>",unsafe_allow_html=True)
        op1 = st.selectbox('Select Graph',('---','Bar chart'))
        
        if op1 == '---':
            st.empty()
        elif op1 == 'Bar chart':
            st.image("Clustering/complete_df/kmode count.png")
        else:
            st.warning("Please select a chart !")
        st.markdown("<h4 style='text-align:center;'>Interpretation<h4>",unsafe_allow_html=True)
        ################################################
        st.markdown("<hr>",unsafe_allow_html=True) 
        op1 = st.selectbox('Visualize predicted clusters By',('---','Age Bins','Category','Is Discount','Region'))
        c1,c2 = st.columns([2,1])
        if op1 == '---':
            st.empty()
        elif op1 == 'Age Bins':
            with c1:
                st.image("Clustering/complete_df/kmode agebins.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In cluster 0, 3, 4 & 6 most
                        of the orders are from age 45 -70.
                        In cluster 1, 5, 7, & 8 it is
                        from age 30 -45. In cluster 2
                        & 9 it is from age 19 - 30 .
                        ''', unsafe_allow_html=True)
                
        elif op1 == 'Category':
            with c1:
                st.image("Clustering/complete_df/kmode category.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                            In cluster 0 & 1 most of
                            the orders are from
                            'Mobiles & Tablets'.
                        ''', unsafe_allow_html=True)
        elif op1 == 'Is Discount': 
            with c1:
                st.image("Clustering/complete_df/kmode isdiscount.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In clsuter 1, 3, 5 & 6 
                        most of the customers have 
                        ordered discounted orders.
                        ''', unsafe_allow_html=True)
        elif op1 == 'Region': 
            with c1:
                st.image("Clustering/complete_df/kmode Regions.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In cluster 0 & 8 most
                        of the orders are from
                        sothern region.
                        ''', unsafe_allow_html=True)
        else:
            st.warning("Please select a chart !")
    elif op == 'Order Canceled Data':
        st.markdown("<h4 style='text-align:'left'>Order Canceled Data",unsafe_allow_html=True) 
        c1, c2 = st.columns(2)
        with c1:
            st.image("Clustering/cancel_df/kmode_elbow_cancel.png")
        with c2:
            
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Results
                    Optimal number of K is 10.
                    ''', unsafe_allow_html=True)
                
        st.markdown("<hr>",unsafe_allow_html=True)
        op1 = st.selectbox('Select Graph',('---','Bar chart'))
        
        if op1 == '---':
            st.empty()
        elif op1 == 'Bar chart':
            st.image("Clustering/cancel_df/kmode count.png")
        else:
            st.warning("Please select a chart !")
        st.markdown("<h4 style='text-align:center;'>Interpretation<h4>",unsafe_allow_html=True)
        ################################################
        st.markdown("<hr>",unsafe_allow_html=True) 
        op1 = st.selectbox('Visualize predicted clusters By',('---','Age Bins','Category','Is Discount','Region'))
        c1,c2 = st.columns([2,1])
        if op1 == '---':
            st.empty()
        elif op1 == 'Age Bins':
            with c1:
                st.image("Clustering/cancel_df/kmode agebins.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In cluster1, 3 & 5 most of the
                        orders are from age 30 -45.
                        In cluster 0, 4 & 8 most
                        of the orders are from  age
                        45 - 70. In cluster 2, 6 & 9
                        most of the orders are fromn age
                        19- 30. Only in cluster 7 most
                        of the orders are from age
                        70 - 100. 
                        ''', unsafe_allow_html=True)
                
        elif op1 == 'Category':
            with c1:
                st.image("Clustering/cancel_df/kmode category.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In cluster 0 & 1 most of the 
                        canceled orders are from 
                        'Mobiles & Tablets'.
                        ''', unsafe_allow_html=True)
        elif op1 == 'Is Discount': 
            with c1:
                st.image("Clustering/cancel_df/kmode isdiscount.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        Only in cluster 1, 8 & 9 most
                        of the canceled orders are 
                        discounted.
                        ''', unsafe_allow_html=True)
        elif op1 == 'Region': 
            with c1:
                st.image("Clustering/cancel_df/kmode Regions.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        Only in cluster 0 & 7 most of 
                        the orders are canceled from 
                        southern region.
                        ''', unsafe_allow_html=True)
        else:
            st.warning("Please select a chart !")
    else:
         st.warning("Please Select Data !")
    
    #Agglomerative
elif option2 == 'Agglomerative':
    st.markdown("<h3 style='text-align:'left'>Agglomerative Clustering</h3>",unsafe_allow_html=True)
    op = st.selectbox('Select Data',('---','Order Completed Data','Order Canceled Data'))
    if op == '---':
        st.empty()
    elif op == 'Order Completed Data':
        st.markdown("<h4 style='text-align:'left'>Order Completed Data",unsafe_allow_html=True) 
        c1, c2 = st.columns(2)
        with c1:
            st.image("Clustering/complete_df/agg_dend.png")
        with c2:
            
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Interpretation
                    Optimal number of K is 6.
                    ''', unsafe_allow_html=True)
                
        st.markdown("<hr>",unsafe_allow_html=True)
        op1 = st.selectbox('Select Graph',('---','Scatter plot','Snake plot'))     
        if op1 == '---':
            st.empty()
        elif op1 == 'Scatter plot':
            st.image("Clustering/complete_df/agg_3d.png")
        elif op1 == 'Snake plot':
            st.image("Clustering/complete_df/agg_snake.png")
        else:
            st.warning("Please select a chart !")
          
        ################################################
        st.markdown("<hr>",unsafe_allow_html=True) 
        op1 = st.selectbox('Visualize predicted clusters By',('---','Category','Gender','Region'))
        c1,c2 = st.columns([2,1])
        if op1 == '---':
            st.empty()
        elif op1 == 'Category':
            with c1:
                st.image("Clustering/complete_df/agg_category.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        Only in cluster 3 & 4 most
                        of the orders are from 
                        'Mobiles & Tablets'. Other 
                        clusters it is 'Mens Fashion'.
                        ''', unsafe_allow_html=True)
                
        elif op1 == 'Gender':
            with c1:
                st.image("Clustering/complete_df/agg_gender.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        Both males and females have orderd
                        nearly in each cluster.
                        ''', unsafe_allow_html=True)
    
        elif op1 == 'Region': 
            with c1:
                st.image("Clustering/complete_df/agg_region.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In all the clusters most of 
                        the orders are from south. 
                        ''', unsafe_allow_html=True)
        else:
            st.warning("Please select a chart !")
    elif op == 'Order Canceled Data':
        st.markdown("<h4 style='text-align:'left'>Order Canceled Data",unsafe_allow_html=True) 
        c1, c2 = st.columns(2)
        with c1:
            st.image("Clustering/cancel_df/agg_dend.png")
        with c2:
            
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Interpretation
                    Optimal number of K is 6.
                    ''', unsafe_allow_html=True)
                
        st.markdown("<hr>",unsafe_allow_html=True)
        op1 = st.selectbox('Select Graph',('---','Scatter plot','Snake plot'))     
        if op1 == '---':
            st.empty()
        elif op1 == 'Scatter plot':
            st.image("Clustering/cancel_df/agg_3d.png")
        elif op1 == 'Snake plot':
            st.image("Clustering/cancel_df/agg_snake.png")
        else:
            st.warning("Please select a chart !")
        
        ################################################
        st.markdown("<hr>",unsafe_allow_html=True) 
        op1 = st.selectbox('Visualize predicted clusters By',('---','Category','Gender','Region'))
        c1,c2 = st.columns([2,1])
        if op1 == '---':
            st.empty()
        elif op1 == 'Category':
            with c1:
                st.image("Clustering/cancel_df/agg_category.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In cluster 1 & 3 most of 
                        the canceled orders are from 'Others'. 
                        ''', unsafe_allow_html=True)
                
        elif op1 == 'Gender':
            with c1:
                st.image("Clustering/cancel_df/agg_gender.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In all the clusters both males
                        and females have canceled nearly 
                        equal number of orders.
                        ''', unsafe_allow_html=True)
    
        elif op1 == 'Region': 
            with c1:
                st.image("Clustering/cancel_df/agg_region.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In all the regions most of the 
                        canceled oreders from south. 
                        ''', unsafe_allow_html=True)
        else:
            st.warning("Please select a chart !")
    else:
         st.warning("Please Select Data !")
    
    #Gaussian Mixture
elif option2 == 'Gaussian Mixture':
    st.markdown("<h3 style='text-align:'left'>Gaussian Mixture</h3>",unsafe_allow_html=True)
    op = st.selectbox('Select Data',('---','Order Completed Data','Order Canceled Data'))
    if op == '---':
        st.empty()
    elif op == 'Order Completed Data':
        st.markdown("<h4 style='text-align:'left'>Order Completed Data",unsafe_allow_html=True) 
        c1, c2 = st.columns([2,1])
        with c1:
            st.image("Clustering/complete_df/gmm_optimal.png")
        with c2:
            
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Results
                    Optimal number of K is 7.
                    ''', unsafe_allow_html=True)
                
        st.markdown("<hr>",unsafe_allow_html=True)
        op1 = st.selectbox('Select Graph',('---','Scatter plot','Snake plot'))     
        if op1 == '---':
            st.empty()
        elif op1 == 'Scatter plot':
            st.image("Clustering/complete_df/gmm_3d.png")
        elif op1 == 'Snake plot':
            st.image("Clustering/complete_df/gmm_snake.png")
        else:
            st.warning("Please select a chart !")
        c1, c2 = st.columns(2)
       
        st.markdown("<hr>",unsafe_allow_html=True) 
        op1 = st.selectbox('Visualize predicted clusters By',('---','Region','Category','Gender'))
        
        if op1 == '---':
            st.empty()
        elif op1 == 'Region':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/complete_df/gmm_region.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In all the clusters most of the 
                        orders are from south. 
                        ''', unsafe_allow_html=True) 
        elif op1 == 'Category':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/complete_df/gmm_category.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In cluster 1, 2 & 3 most of the
                        orders are from 'Mens Fashion'.
                        All the other clusters except
                        cluster 5 it is 'Mobiles & Tablets'.
                        In cluster 5 it is 'Woman Fashion'.
                        ''', unsafe_allow_html=True)
        elif op1 == 'Gender':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/complete_df/gmm_Gender.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        Only in cluster 1,5 & 6 most
                        of the customers are Females.
                        ''', unsafe_allow_html=True)
        else:
            st.warning("Please select a chart !")
            
        
    elif op == 'Order Canceled Data':
        st.markdown("<h4 style='text-align:'left'>Order Canceled Data",unsafe_allow_html=True) 
        c1, c2 = st.columns([2,1])
        with c1:
            st.image("Clustering/cancel_df/gmm_optimal.png")
        with c2:
            
            with st.expander('', expanded=True):
                st.markdown(f'''
                    ##### Results
                    Optimal number of K is 9.
                    ''', unsafe_allow_html=True)
                
        st.markdown("<hr>",unsafe_allow_html=True)
        op1 = st.selectbox('Select Graph',('---','Scatter plot','Snake plot'))     
        if op1 == '---':
            st.empty()
        elif op1 == 'Scatter plot':
            st.image("Clustering/cancel_df/gmm_3d.png")
        elif op1 == 'Snake plot':
            st.image("Clustering/cancel_df/gmm_snake.png")
        else:
            st.warning("Please select a chart !")
        c1, c2 = st.columns(2)
       
        st.markdown("<hr>",unsafe_allow_html=True) 
        op1 = st.selectbox('Visualize predicted clusters By',('---','Region','Category','Gender'))
        
        if op1 == '---':
            st.empty()
        elif op1 == 'Region':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/cancel_df/gmm_region.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In all the clusters most of the 
                        canceled orders are from south. 
                        ''', unsafe_allow_html=True) 
        elif op1 == 'Category':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/cancel_df/gmm_category.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In cluster 1, 2, 4 & 5 most 
                        of the canceled orders are from 
                        'Mobiles & Tablets'.  
                        ''', unsafe_allow_html=True)
        elif op1 == 'Gender':
            c1,c2 = st.columns([2,1])
            
            with c1:
                st.image("Clustering/cancel_df/gmm_Gender.png")
            with c2:
                with st.expander('', expanded=True):
                    st.markdown(f'''
                        ##### Results
                        In all the clusters nearly equal 
                        number of orders are canceled by 
                        both males and females.
                        ''', unsafe_allow_html=True)
        else:
            st.warning("Please select a chart !")
    else:
        st.warning("Please Select Data !")
else:
    st.warning("Please select Clustering technique !")
    
if (options1 == '---') & (option2 == '---') & (variable == '---'):
    st.markdown("<h3 class = 'subheader'> DSC3263 : Independent Study in Data Science</h3>",unsafe_allow_html=True)
    div1, div2 = st.columns([2,1])
    div1.image("pics/segmentation.png",width=500)
    div2.markdown("<div class = 'about'> <h4 class = 'group'>Group 3</h4><ul class = 'mem'><h6 class = 'member'>Sumedha Kulasekara (S/17/403)</h6><h6 class = 'member'>Lasantha Kulasooriya (S/17/404)</h6></ul></div>",unsafe_allow_html=True)  
    
