import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns

import helper
import preprocessor
import style

import plotly.express as px
import plotly.figure_factory as ff

df = pd.read_csv("athlete_events.csv")
region_df = pd.read_csv("noc_regions.csv")

st.sidebar.title("Olympics Analysis (Till 2016)")

user_option = st.sidebar.radio(
    'Select a Option',
    ['Overall Analysis', 'Medal Tally', 'Country-wise Analysis', 'Athlete wise Analysis']
)

st.sidebar.header(user_option)
df = preprocessor.preprocess(df, region_df)

years, regions, seasons, sports_list = helper.get_selectbox_options(df)

css = style.css
st.markdown(css, unsafe_allow_html=True)

if user_option == "Medal Tally":
    year = st.sidebar.selectbox(
        'Select a Year', years
    )

    country = st.sidebar.selectbox(
        'Select a Country', regions
    )

    season = st.sidebar.selectbox(
        'Select a Season', seasons
    )
    medal_tally, heading = helper.get_medal_tally(df, country, year, season)
    st.header(heading)
    st.table(medal_tally)

    st.header("Most Successful Athletes")
    sport = st.selectbox("Select a Sport", sports_list)
    successful_athlete = helper.get_successful_athletes(df, country, year, season, sport)
    st.table(successful_athlete)

if user_option == "Overall Analysis":
    season = st.sidebar.selectbox(
        'Select a Season', seasons
    )

    editions, hosts, sports, events, athletes, nations, heading = helper.get_top_stats(df, season)

    st.header(heading)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Editions")
        st.title(editions)

    with col2:
        st.subheader("Hosts")
        st.title(hosts)

    with col3:
        st.subheader("Nations")
        st.title(nations)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader("Events")
        st.title(events)

    with col5:
        st.subheader("Sports")
        st.title(sports)

    with col6:
        st.subheader("Athletes")
        st.title(athletes)

    country_over_time = helper.get_country_participation(df, season)
    fig = px.line(country_over_time, x="Year", y="No of Countries")
    st.header("Country Participation over the Years")
    st.plotly_chart(fig)

    events_over_time = helper.get_event_participation(df, season)
    fig = px.line(events_over_time, x="Year", y="No of Events")
    st.header("No of Events over the Years")
    st.plotly_chart(fig)

    athletes_over_time = helper.get_athletes_participation(df, season)
    fig = px.line(athletes_over_time, x="Year", y="Total", color='Sex', title="Male Vs Female")
    st.header("Athlete Participation over the Years")
    st.plotly_chart(fig)

    st.header("No of Events over the Years(By Sport)")
    fig, ax = plt.subplots(figsize=(20, 20))
    sport_df, heading = helper.fetch_df_top_stats(df, season)
    sport_df = sport_df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(sport_df.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0)
                     .astype('int'), annot=True)
    st.pyplot(fig)

if user_option == "Country-wise Analysis":
    season = st.sidebar.selectbox(
        'Select a Season', seasons
    )

    country = st.sidebar.selectbox(
        'Select a Country', regions
    )

    country_medal_tally = helper.get_yearwise_medal_tally_per_country(df, season, country)
    fig = px.line(country_medal_tally, x="Year", y="Medal")
    st.header("No of Medals over the Years")
    st.plotly_chart(fig)

    sportwise_country_medals, heading = helper.get_sportwise_country_heatmap(df, season, country)
    st.header(heading)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(sportwise_country_medals, annot=True)
    st.pyplot(fig)

    country_wise_best_athletes = helper.get_country_wise_best_athletes(df, season, country)
    st.header("Best performing athletes")
    st.table(country_wise_best_athletes)

if user_option == "Athlete wise Analysis":
    st.header("Distribution of Age")
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Gold'] == 1]['Age'].dropna()
    x3 = athlete_df[athlete_df['Silver'] == 1]['Age'].dropna()
    x4 = athlete_df[athlete_df['Bronze'] == 1]['Age'].dropna()
    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                             show_hist=False, show_rug=False)
    fig.update_layout(xaxis_title='Age', yaxis_title='Medals',autosize=False,width=1000,height=600)
    st.plotly_chart(fig)
