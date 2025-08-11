import os
import fitz  # PyMuPDF: to read and parse PDF files
import pandas as pd
import numpy as np
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set the path where PDFs are stored
#output_folder = "/Users/AlbertNadar/Desktop/PdfExtracts"
output_folder = "PdfExtracts"
def readpdftodf():
    all_data = []  # To collect data from all PDFs

    # Loop over each file in the folder
    for filename in os.listdir(output_folder):
        if filename.lower().endswith(".pdf"):  # Process only PDFs
            pdf_path = os.path.join(output_folder, filename)

            doc = fitz.open(pdf_path)  # Open PDF file
            transaction_rows = []  # Store potential transaction rows

            for page in doc:
                blocks = page.get_text("blocks")  # Extract text blocks from page
                blocks = sorted(blocks, key=lambda b: b[1])  # Sort blocks top-to-bottom using vertical y0

                for b in blocks:
                    text = b[4].strip()  # Extract the actual text from the block
                    if any(c.isdigit() for c in text) and "/" in text:
                        # Heuristic: Line contains a digit and slash → likely a transaction date line
                        transaction_rows.append(text)

            if not transaction_rows:
                print(f"No transaction rows found in {filename}")
                continue  # Skip to next file

            # Split rows based on two or more spaces (column separator)
            data = [re.split(r'\s{2,}', row) for row in transaction_rows]

            # Ensure uniform column count
            max_cols = max(len(row) for row in data)
            for row in data:
                row += [''] * (max_cols - len(row))

            df = pd.DataFrame(data)
            df.columns = ['Date', 'SerNo', 'Description', 'Points', 'AmountFlag', 'Amount'][:df.shape[1]]

            # This line flattens multi-line dates into separate rows (if they exist), but only keeps "Date"
            df = df['Date'].str.split('\n', expand=True)

            def is_number(x):
                try:
                    float(x)
                    return True
                except:
                    return False

            df = df.iloc[:, :6]  # Trim extra columns if any

            # Check if the 'Points' column is numeric to identify misaligned columns
            df['IsNumber'] = df.iloc[:, 3].apply(is_number)

            # Fix 'Description' if split incorrectly due to misalignment
            df.iloc[:, 2] = df.apply(
                lambda x: str(x[2]) + ' ' + str(x[3]) if not x['IsNumber'] else x[2], axis=1
            )
            df.iloc[:, 3] = df.apply(lambda x: x[3] if x['IsNumber'] else 0, axis=1)

            # Rename columns for clarity
            df.columns = ['Date', 'LineID', 'Name', 'Cashback', 'Amount', 'Unknown', 'IsNumber']

            # Fix 'Cashback' and 'Amount' if columns were misaligned
            df['Cashback'] = df.apply(
                lambda x: x['Amount'] if (not x['IsNumber']) and pd.notnull(x['Unknown']) else x['Cashback'],
                axis=1
            )
            df['Amount'] = df.apply(
                lambda x: x['Unknown'] if (not x['IsNumber']) and pd.notnull(x['Unknown']) else x['Amount'],
                axis=1
            )

            # Remove helper columns
            df.drop(columns=['Unknown', 'IsNumber'], inplace=True)

            # Convert 'Date' to datetime, filter out invalid ones
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
            df = df[df['Date'].notna()]  # Keep only rows with valid dates

            # Mark whether transaction is Credit or Debit
            df['Type'] = np.where(df['Amount'].astype(str).str.contains('CR'), 'CR', 'DB')

            # Clean amount formatting
            df['Amount'] = df['Amount'].astype(str).str.replace(' CR', '', regex=False)
            df['Amount'] = df['Amount'].astype(str).str.replace(',', '', regex=False)

            # Remove known non-transaction rows
            df = df.query("Name != 'BBPS Payment received'")
            df.reset_index(drop=True, inplace=True)
            df = df.sort_values(by=['Date', 'LineID'])

            # Convert to proper data types
            df['Name'] = df['Name'].astype('string')
            df['LineID'] = df['LineID'].astype(int)
            df['Cashback'] = df['Cashback'].astype(float)
            df['Amount'] = df['Amount'].astype(float)
            df['Type'] = df['Type'].astype('string')

            # ---------- Tagging Logic ----------
            tag_rules = [
                ('interest amount amortization', 'EMI'),
                ('principal amount amortization', 'EMI'),
                ('sgst-ci@9%', 'EMI'),
                ('cgst-ci@9%', 'EMI'),
                ('pharmeasy', 'Medical'),
                ('AXELIA SOLUTIONS PVT LTD','Medical'),
                ('reliance jio', 'Phone'),
                ('royal service mumbai in', 'Petrol'),
                ('PETROLEUM','Petrol'),
                ('ICARPORT','Petrol'),
                ('swiggy', 'Food'),
                ('zomato', 'Food'),
                ('INSTAMART','Food'),
                ('http://www.am', 'Amazon'),
                ('Reversal of Fuel Surcharge', 'Petrol Reversal'),
                ('SGST-Rev-CI@9%', 'Petrol Reversal'),
                ('CGST-Rev-CI@9%', 'Petrol Reversal'),
                ('NETFLIX','Movies'),
                ('MSW*SHAMAN MOTORS PRIVA ','Bike Service'),
                ('CLEARTRIP','Travel'),
                ('IBIBO GROUP PVT LTD','Travel'),
                ('IRCTC ','Train'),
                ('Medicare','Hospital'),
                ('EYEWEARLABS','Shopping'),
                ('DMART AVENUE SUPERMART','Grocery')
            ]

            # Apply tags based on description matches
            def taglogic(x):
                value = str(x['Name']).strip().lower()
                for keyword, tag in tag_rules:
                    if keyword.lower() in value:
                        return tag
                return 'Others'

            df['Tag'] = df.apply(taglogic, axis=1)

            # Flip sign of credit amounts to negative
            df['Amount'] = df.apply(lambda x: -x['Amount'] if x['Type'] == 'CR' else x['Amount'], axis=1)

            # Add metadata columns
            df['StatementMonth'] = df['Date'].max().strftime('%B %Y') if not df.empty else None
            df['CardNo'] = filename
            df['CardNo'] = df['CardNo'].str[:4]

            # this is created to add current month 
            df['CurrentDateMonth'] = datetime.date.today() - relativedelta(months=1)
            df['CurrentDateMonth'] = df['CurrentDateMonth'].apply(lambda x: x.strftime('%B %Y'))

            # this is created to add previous month
            df['PreviousDateMonth'] = datetime.date.today() - relativedelta(months=2)
            df['PreviousDateMonth'] = df['PreviousDateMonth'].apply(lambda x: x.strftime('%B %Y'))

            # Append to master list
            all_data.append(df)

    # Combine all processed DataFrames
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        print(f"Total rows: {len(final_df)}")
        return final_df
    else:
        print("No data collected from any file.")
        return pd.DataFrame()

#run the function
df = readpdftodf()

df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Type'] == 'DB']  # Filter Type = DB

# ----------- 2. Header ---------------------------
st.set_page_config(layout="wide")
st.title("Spend Analysis Dashboard")

# ----------- 3. Monthly Revenue Chart -------------
monthly_revenue = df.resample('ME', on='Date')['Amount'].sum().reset_index()
monthly_revenue['Month'] = monthly_revenue['Date'].dt.strftime('%b-%Y')
last_8 = monthly_revenue.tail(8)

fig_bar = px.bar(
    last_8,
    x='Month',
    y='Amount',
    title='Spend by Month (Last 8)',
    text='Amount',
    color='Amount',
    color_continuous_scale=px.colors.sequential.Blues
)
fig_bar.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
fig_bar.update_layout(yaxis_title="Total Spend", xaxis_title="Month", height=500)

st.plotly_chart(fig_bar, use_container_width=True)

cols2 = st.columns(3)
## For top 10 current month
toptencurrentmonth=df[df['CurrentDateMonth'] == df['StatementMonth']].query("Type == 'DB'").sort_values(by='Amount',ascending=False).head(10)
toptencurrentmonth = toptencurrentmonth.head(10)[['Date','Tag','Amount','Cashback']].reset_index(drop=True)
toptencurrentmonth['Date'] = pd.to_datetime(toptencurrentmonth['Date']).dt.date

# Apply bar-style visualization to 'Amount' column
styled_table = toptencurrentmonth.style.bar(
    subset=['Amount'], color='#66b3ff'  # You can match your bar chart color here
).format({'Amount': '₹{:,.0f}', 'Cashback': '₹{:,.0f}'})  # Optional: add formatting

# Display in first column
cols2[2].dataframe(styled_table, use_container_width=True)

tag_all = toptencurrentmonth.groupby('Tag')['Amount'].sum().reset_index()

fig_pie_all2 = go.Figure(data=[
    go.Pie(
        labels=tag_all['Tag'],
        values=tag_all['Amount'],
        hole=0.4,
        textinfo='percent+label',
        marker=dict(colors=px.colors.sequential.Blues[::-1])
    )
])
fig_pie_all2.update_layout(title_text='Current Month Spend Distribution by Tag', height=400)


cols2[1].plotly_chart(fig_pie_all2, use_container_width=True)

CMAmount = df[df['CurrentDateMonth'] == df['StatementMonth']].query("Type == 'DB'")['Amount'].sum()
CMCashback = df[df['CurrentDateMonth'] == df['StatementMonth']].query("Type == 'DB'")['Cashback'].sum()
AvgAmount = df['Amount'].mean()
PMAmount = df[df['PreviousDateMonth'] == df['StatementMonth']].query("Type == 'DB'")['Amount'].sum()
PMCashback = df[df['PreviousDateMonth'] == df['StatementMonth']].query("Type == 'DB'")['Cashback'].sum()
DeltaAmount = CMAmount - PMAmount
DeltaCashback = CMCashback - PMCashback


# #cols = st.columns(3)
# cols2[0].metric("Total Amount", f"₹{CMAmount:,.0f}", f"{DeltaAmount:+,}", delta_color="normal")
# cols2[0].metric("Total Cashback", f"₹{CMCashback:,.0f}", f"{DeltaCashback:+,}", delta_color="inverse")
# cols2[0].metric("Average Amount", f"₹{AvgAmount:,.0f}")

with cols2[0]:
    with st.container():
        st.subheader("📈 Monthly KPIs")
        st.markdown("<div style='background-color:#f0f8ff; padding:15px; border-radius:10px'>", unsafe_allow_html=True)
        #st.metric("💰 Total Amount", f"₹{CMAmount:,.0f}", f"{DeltaAmount:+,}", delta_color="normal")
        arrow = "🔼" if DeltaAmount > 0 else "🔽"
        st.metric(f"💰 Total Amount {arrow}", f"₹{CMAmount:,.0f}", f"{DeltaAmount:+,}")
        st.markdown("</div>", unsafe_allow_html=True)
        #st.markdown("**🛈 Note:** Compared to previous month", unsafe_allow_html=True)

with cols2[0]:
    with st.container():
        st.markdown("<div style='background-color:#e6ffe6; padding:15px; border-radius:10px'>", unsafe_allow_html=True)
        st.metric("🎁 Total Cashback", f"₹{CMCashback:,.0f}", f"{DeltaCashback:+,}", delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)

with cols2[0]:
    with st.container():
        st.markdown("<div style='background-color:#fff0f5; padding:15px; border-radius:10px'>", unsafe_allow_html=True)
        st.metric("📊 Average Amount", f"₹{AvgAmount:,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------- 4. Pie Chart - All Time -------------
st.subheader("Spend Breakdown by Tag")

cols1 = st.columns(([1, 1, 2]))
tag_all = df.groupby('Tag')['Amount'].sum().reset_index()

fig_pie_all = go.Figure(data=[
    go.Pie(
        labels=tag_all['Tag'],
        values=tag_all['Amount'],
        hole=0.4,
        textinfo='percent+label',
        marker=dict(colors=px.colors.sequential.Blues[::-1])
    )
])
fig_pie_all.update_layout(title_text='Overall Spend Distribution by Tag', height=400)

cols1[0].plotly_chart(fig_pie_all, use_container_width=True)

## this is for Cashback
tag_cashback = df.query('Cashback > 0').groupby('Tag')['Cashback'].sum().reset_index()

fig_pie_cashback = go.Figure(data=[
    go.Pie(
        labels=tag_cashback['Tag'],
        values=tag_cashback['Cashback'],
        hole=0.4,
        textinfo='percent+label',
        marker=dict(colors=px.colors.sequential.Blues[::-1])
    )
])
fig_pie_cashback.update_layout(title_text='Cashback by Tag', height=400)

cols1[1].plotly_chart(fig_pie_cashback, use_container_width=True)

## for heat Map
df['Weekday'] = df['Date'].dt.day_name()
df['Month'] = df['Date'].dt.strftime('%b-%Y')

heatmap_data = df.groupby(['Month', 'Weekday'])['Amount'].sum().reset_index()

fig_heat = px.density_heatmap(
    heatmap_data,
    x='Weekday', y='Month', z='Amount',
    color_continuous_scale='Blues',
    title="Weekday vs Month Spend Heatmap"
)
cols1[2].plotly_chart(fig_heat, use_container_width=True)

# ----------- 5. Pie Charts - Last 3 Months -------------
st.subheader("Last 3 Months - Tag Wise Spend")

# Extract last 3 months as Periods
last_3_months = sorted(df['Date'].dt.to_period('M').unique())[-3:]

cols = st.columns(3)
for i, period in enumerate(last_3_months):
    month_df = df[df['Date'].dt.to_period('M') == period]
    tag_month = month_df.groupby('Tag')['Amount'].sum().reset_index()

    fig = go.Figure(data=[
        go.Pie(
            labels=tag_month['Tag'],
            values=tag_month['Amount'],
            hole=0.4,
            textinfo='percent+label',
            marker=dict(colors=px.colors.sequential.Blues[::-1])
        )
    ])
    fig.update_layout(title_text=f"{period.strftime('%b-%Y')}", height=400)

    cols[i].plotly_chart(fig, use_container_width=True)
