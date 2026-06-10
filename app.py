import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

#page configuration
st.set_page_config(
    page_title="Afficionado Coffee - Demad Forecaster",
    page_icon="☕",
    layout="wide"
)

#data load
@st.cache_data
def load_data():
    daily = pd.read_csv(r'C:\Users\hp\afficionado_forecasting\data\daily_features.csv')
    daily['date'] = pd.to_datetime(daily['date'])

    models = pd.read_csv(r'C:\Users\hp\afficionado_forecasting\data\model_results.csv')
    scenarios = pd.read_csv(r'C:\Users\hp\afficionado_forecasting\data\scenarios.csv')
    peaks = pd.read_csv(r'C:\Users\hp\afficionado_forecasting\data\peak_hours.csv')

    raw = pd.read_csv(r'C:\Users\hp\afficionado_forecasting\data\transactions.csv')
    raw['transaction_time'] = pd.to_datetime(raw['transaction_time'], format='%H:%M:%S')
    raw['hour'] = raw['transaction_time'].dt.hour
    raw['revenue'] = raw['transaction_qty'] * raw['unit_price']

    return daily, models, scenarios, peaks, raw
daily, models, scenarios, peaks, raw = load_data()

#sidebar

st.sidebar.title("Afficionado Coffee")
st.sidebar.markdown("**Demand Forecasting Dashboard**")
st.sidebar.markdown("---")

store = st.sidebar.selectbox(
    "Select Store",
    options = daily['store_location'].unique()
)

horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7, max_value=30, value=14, step=7
)

target = st.sidebar.radio(
    "Forecast Target",
    options=["Revenue ($)", "Transaction Volume"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data period:** Jan-Jun 2025")
st.sidebar.markdown("**Total transactions:** 146,116")
st.sidebar.markdown("**Stores:** 3 NYC locations")

#header
st.title("☕ Afficionado Coffee Roasters")
st.subheader("Data-Driven Demand Forecasting & Peak Demand Prediction")
st.markdown("---")

#KPI Cards
store_data = daily[daily['store_location'] == store]
total_revenue = store_data['total_revenue'].sum()
avg_daily = store_data['total_revenue'].mean()
max_daily = store_data['total_revenue'].max()
best_model = models[models['store'] == store].sort_values('MAPE').iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue (6mo)", f"${total_revenue:,.0f}")
col2.metric("Avg Daily Revenue", f"${avg_daily:,.0f}")
col3.metric("Peak Day Revenue", f"${max_daily:,.0f}")
col4.metric("Best Model MAPE", f"${best_model['MAPE']:.1f}%",
            delta=f"{best_model['model']}", delta_color="off")

st.markdown("---")

#tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast", "🔥 Peak Hours", "📊 Model Comparison", "🎯 Scenarios"
])

#TAB1: forecast
with tab1:
    st.subheader(f"{store} - {horizon}-Day Revenue Forecast")

    @st.cache_data
    def run_prophet(store_name, horizon):
        df_store = daily[daily['store_location'] == store_name][['date','total_revenue']].rename(
            columns={'date':'ds','total_revenue':'y'}
        )

        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        m.fit(df_store)
        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)
        return df_store, forecast
    
    df_store, forecast = run_prophet(store, horizon)

    #historical and future split
    last_date = df_store['ds'].max()
    hist_fc = forecast[forecast['ds'] <= last_date]
    fut_fc = forecast[forecast['ds'] > last_date]

    fig = go.Figure()

    #actual revenue
    fig.add_trace(go.Scatter(
        x=df_store['ds'], y=df_store['y'],
        name='Actual Revenue', line=dict(color='#333333', width=1.5)
    ))

    #forecast line
    fig.add_trace(go.Scatter(
        x=fut_fc['ds'], y=fut_fc['yhat'],
        name='Forecast', line=dict(color='#E8593C', width=2, dash='dash')
    ))

    #confidence interval
    fig.add_trace(go.Scatter(
        x = pd.concat([fut_fc['ds'], fut_fc['ds'][::-1]]),
        y = pd.concat([fut_fc['yhat_upper'], fut_fc['yhat_lower'][::-1]]),
        fill='toself', fillcolor= 'rgba(232,89,60,0.15)',
        line = dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))

    fig.update_layout(
        xaxis_title='Date', yaxis_title= 'Revenue ($)',
        legend=dict(orientation='h', y=1.1),
        height=400, margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig, width='stretch', key='forecast_main')

    #forecast table
    st.markdown("**Upcoming forecasted revenue:**")
    fc_table = fut_fc[['ds','yhat','yhat_lower','yhat_upper']].copy()
    fc_table.columns = ['Date','Expected ($)','Low ($)', 'High ($)']
    fc_table['Date'] = fc_table['Date'].dt.strftime('%Y-%m-%d')
    fc_table[['Expected ($)','Low ($)', 'High ($)']] = fc_table[
        ['Expected ($)','Low ($)', 'High ($)']].round(0).astype(int)
    st.dataframe(fc_table, width='stretch', hide_index=True)


#TAB2: peak hours
with tab2:
    st.subheader(f"{store} - Hourly Demand Heatmap")

    store_raw = raw[raw['store_location'] == store]
    hourly_counts = store_raw.groupby('hour')['transaction_id'].count().reset_index()
    hourly_counts.columns = ['Hour', 'Transactions']

    #bar chart
    peak_threshold = hourly_counts['Transactions'].quantile(0.75)
    hourly_counts['is_peak'] = hourly_counts['Transactions'] >= peak_threshold

    fig2 = px.bar(
        hourly_counts, x='Hour', y='Transactions',
        color='is_peak',
        color_discrete_map={True:'#E8593C', False:'#A0836B'},
        labels={'is_peak': 'Peak Hour'},
        title=f'Transaction Volume by Hour - {store}'
    )
    fig2.update_layout(height=350, showlegend=True,
                       margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig2, width='stretch',key="peak_bar")

    peak_hours = hourly_counts[hourly_counts['is_peak']]['Hour'].tolist()
    st.info(f"⚡ **Peak staffing hours for {store}:** {peak_hours[0]}:00 – {peak_hours[-1]+1}:00 "
            f"| Recommend maximum staff during this window ")

    #store comparison
    st.markdown("**All Stores - Peak Hour Comparison**")
    all_hourly = raw.groupby(['store_location','hour'])['transaction_id'].count().reset_index()
    all_hourly.columns = ['Store','Hour','Transactions']

    fig3 = px.line(all_hourly, x='Hour', y='Transactions',
                   color='Store', markers=True,
                   title='Hourly Transaction Volume - All Stores')
    fig3.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, width='stretch',key="stores")

#TAB3 : Model comparison
with tab3:
    st.subheader("Model Performance Comparison")
    col1, col2= st.columns(2)

    with col1:
        st.markdown("**All Models - MAE by Store**")
        fig4 = px.bar(
            models, x='store', y='MAE',
            color='model', barmode='group',
            color_discrete_sequence=['#6F4E37','#E8593C','#A0C878','#D4A853']
        )
        fig4.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig4, width='stretch',key="model_mae")

    with col2:
        st.markdown("**All Models - MAPE (%) by Store**")
        fig5 = px.bar(
            models, x='store', y='MAPE',
            color='model', barmode='group',
            color_discrete_sequence=['#6F4E37','#E8593C','#A0C878','#D4A853']
        )
        fig5.update_layout(height=350,  margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig4, width='stretch',key="model_mape")
    
    st.markdown("**Full Results Table**")
    st.dataframe(
        models.sort_values(['store','MAPE']).round(2),
        width='stretch', hide_index=True
    )

#TAB4 : Scenarios
with tab4:
    st.subheader("30-Day Revenue Scenario Planning")

    fig6 = go.Figure()
    stores_list = scenarios['Store'].tolist()

    fig6.add_trace(go.Bar(
        name='Worst Case', x=stores_list,
        y=scenarios['Worst Case ($)'], marker_color='#D05538'
    ))
    fig6.add_trace(go.Bar(
        name='Expected', x=stores_list,
        y=scenarios['Expected ($)'], marker_color='#6F4E37'
    ))
    fig6.add_trace(go.Bar(
        name='Best Case', x=stores_list,
        y=scenarios['Best Case ($)'], marker_color='#A0C878'
    ))

    fig6.update_layout(
        barmode='group', height=400,
        yaxis_title='Projected Revenue ($)',
        yaxis_tickformat='$,.0f',
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig6, width='stretch',key="sccenarios_bar")

    st.markdown("**Scenario Summary Table**")
    st.dataframe(scenarios, width='stretch', hide_index=True)

    st.markdown("---")
    st.subheader("Demand Spike Calculator")

    spike = st.slider("Simulate demand spike (%)", 0, 100, 30, step=5)

    base_qty = raw[raw['store_location'] == store]['transaction_qty'].sum() / \
              daily[daily['store_location'] == store]['date'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Normal daily transactions", f"{base_qty:.0f}")
    col2.metric(f"At +{spike}% spike",       f"{base_qty * (1 + spike/100):.0f}")
    col3.metric("Extra transactions",        f"+{base_qty * spike/100:.0f}")