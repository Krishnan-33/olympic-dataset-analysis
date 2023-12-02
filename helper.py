import numpy as np
import pandas as pd


def get_selectbox_options(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    region = np.unique(df['region'].dropna().values).tolist()
    region.sort()
    region.insert(0, 'Overall')

    season = df['Season'].unique().tolist()
    season.sort()
    season.insert(0, 'All')

    sports = df['Sport'].unique().tolist()
    sports.sort()
    sports.insert(0, 'Overall')

    return years, region, season, sports


def get_medal_tally(df, country, year, season):
    df, heading = fetch_df_medal_tally(df, country, year, season)

    medal_tally = df.drop_duplicates(
        subset=['Team', 'region', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event', 'Medal'])

    if country != 'Overall' and year == 'Overall':
        medal_tally = medal_tally.groupby('Year').sum(numeric_only=True)[['Gold', 'Silver', 'Bronze']] \
            .sort_values('Year').reset_index()

        medal_tally['Year'] = medal_tally['Year'].apply(lambda x: '{:04d}'.format(x))

        grandtotal = {'Year': 'Grand Total', 'Gold': sum(medal_tally['Gold']), 'Silver': sum(medal_tally['Silver']),
                      'Bronze': sum(medal_tally['Bronze'])}

        grandtotal = pd.DataFrame(grandtotal,index=[0])
        medal_tally = pd.concat([medal_tally, grandtotal], ignore_index=True)
        
    else:
        medal_tally = medal_tally.groupby('region').sum(numeric_only=True)[['Gold', 'Silver', 'Bronze']] \
            .sort_values('Gold', ascending=False).reset_index()

    medal_tally['Total'] = sum([medal_tally['Gold'], medal_tally['Silver'], medal_tally['Bronze']])

    return medal_tally, heading


def get_successful_athletes(df, country, year, season, sport):
    df, heading = fetch_df_medal_tally(df, country, year, season)
    temp_df = df.dropna(subset=['Medal'])

    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]
    medal_tally = temp_df['Name'].value_counts().reset_index().head(15).merge(df, left_on='Name', right_on='Name',
                                                                              how='left').drop_duplicates(
        'Name').reset_index()
    medal_tally.rename(columns={'count': 'Medals'}, inplace=True)
    medal_tally = medal_tally[['Name', 'Medals', 'Sport', 'region']]
    return medal_tally


def fetch_df_medal_tally(df, country, year, season):
    if season != "All":
        df = df[df['Season'] == season]

    if country == 'Overall' and year == 'Overall':
        if season != "All":
            heading = f"Overall Medal Tally {season} Olympics"
        else:
            heading = f"Overall Medal Tally"
        return df, heading

    if country != 'Overall' and year == 'Overall':
        heading = f"Medal Tally for {country}"
        return df[df['region'] == country], heading

    if country == 'Overall' and year != 'Overall':
        if season != "All":
            heading = f"Medal Tally in {year} {season} Olympics"
        else:
            heading = f"Medal Tally in {year} Olympics"
        return df[df['Year'] == year], heading

    if country != 'Overall' and year != 'Overall':
        if season != "All":
            heading = f"Medal Tally in {year} {season} Olympics for {country}"
        else:
            heading = f"Medal Tally in {year} Olympics for {country}"
        return df[(df['region'] == country) & (df['Year'] == year)], heading


def get_top_stats(df, season):
    df, heading = fetch_df_top_stats(df, season)

    editions = df['Year'].unique().shape[0]
    hosts = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    return editions, hosts, sports, events, athletes, nations, heading


def fetch_df_top_stats(df, season):
    heading = "Top Stats"

    if season != "All":
        df = df[df['Season'] == season]
        heading = f"Top Stats {season} Olympics"

    return df, heading


def get_country_participation(df, season):
    df, heading = fetch_df_top_stats(df, season)

    country_over_time = df.drop_duplicates(['Year', 'region'])['Year'].value_counts().reset_index().sort_values('Year')
    country_over_time.rename(columns={'count': 'No of Countries'}, inplace=True)

    return country_over_time


def get_event_participation(df, season):
    df, heading = fetch_df_top_stats(df, season)

    events_over_time = df.drop_duplicates(['Year', 'Event'])['Year'].value_counts().reset_index().sort_values('Year')
    events_over_time.rename(columns={'count': 'No of Events'}, inplace=True)

    return events_over_time


def get_athletes_participation(df, season):
    df, heading = fetch_df_top_stats(df, season)

    athlete = df.drop_duplicates(['Year', 'region', 'Name'])
    athlete = athlete[['Name', 'Sex', 'region', 'Year']]
    athlete = athlete.groupby(['Sex', 'Year'])['Name'].nunique().reset_index()
    athlete = athlete.sort_values('Year')

    total_participants = athlete.groupby('Year')['Name'].sum().reset_index()
    total_participants['Sex'] = 'All'

    merged_dataset = pd.merge(athlete, total_participants, on=['Year', 'Sex'], how='outer').sort_values('Year')
    merged_dataset['Name_x'] = merged_dataset['Name_x'].fillna(merged_dataset['Name_y'])
    merged_dataset.rename(columns={'Name_x': 'Total'}, inplace=True)
    merged_dataset.sort_values(['Year', 'Sex'])
    merged_dataset['Total'] = merged_dataset['Total'].astype('int32')
    return merged_dataset


def get_yearwise_medal_tally_per_country(df, season, country):
    df, heading = fetch_df_top_stats(df, season)
    country_df = df.dropna(subset=['Medal'])
    country_df.drop_duplicates(subset=['Team', 'region', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event', 'Medal'],
                               inplace=True)
    new_df = country_df
    if country != 'Overall':
        new_df = country_df[country_df['region'] == country]

    new_df = new_df.groupby('Year').count()['Medal'].reset_index()

    return new_df


def get_sportwise_country_heatmap(df, season, country):
    df, heading = fetch_df_top_stats(df, season)
    country_df = df.dropna(subset=['Medal'])
    country_df.drop_duplicates(subset=['Team', 'region', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event', 'Medal'],
                               inplace=True)
    new_df = country_df
    if country != 'Overall':
        new_df = country_df[country_df['region'] == country]

    new_df = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0).astype('int')

    heading = f"{country} Progress Sport-Wise"

    return new_df, heading


def get_country_wise_best_athletes(df, season, country):
    df, heading = fetch_df_top_stats(df, season)
    temp_df = df.dropna(subset=['Medal'])
    if country != 'Overall':
        temp_df = temp_df[temp_df['region'] == country]
    medal_tally = temp_df['Name'].value_counts().reset_index().head(15).merge(df, left_on='Name', right_on='Name',
                                                                              how='left').drop_duplicates(
        'Name').reset_index()
    medal_tally.rename(columns={'count': 'Medals'}, inplace=True)
    medal_tally = medal_tally[['Name', 'Medals', 'Sport', 'region']]
    return medal_tally


def get_athlete_medal_distplot(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Gold'] == 1]['Age'].dropna()
    x3 = athlete_df[athlete_df['Silver'] == 1]['Age'].dropna()
    x4 = athlete_df[athlete_df['Bronze'] == 1]['Age'].dropna()
