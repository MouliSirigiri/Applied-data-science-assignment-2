# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 08:00:23 2023

@author: mouli
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read_data(filename):
    df = pd.read_csv(filename, skiprows=4)

    # drop unnecessary columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # rename remaining columns
    df = df.rename(columns={'Country Name': 'Country'})

    # melt the dataframe to convert years to a single column
    df = df.melt(id_vars=['Country', 'Indicator Name'],
                 var_name='Year', value_name='Value')

    # convert year column to integer and value column to float
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # separate dataframes with years and countries as columns
    df_years = df.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # clean the data
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries


def calculate_summary_stats(df_years, countries, indicators):
    # create a dictionary to store the summary statistics
    summary_stats = {}

    # calculate summary statistics for each indicator and country
    for indicator in indicators:
        for country in countries:
            # summary statistics for individual countries
            stats = df_years.loc[(country, indicator)].describe()
            summary_stats[f'{country} - {indicator}'] = stats

        # summary statistics for the world
        stats = df_years.loc[('World', indicator)].describe()
        summary_stats[f'World - {indicator}'] = stats

    return summary_stats


def print_summary_stats(summary_stats):
    # print the summary statistics
    for key, value in summary_stats.items():
        print(key)
        print(value)
        print()


# create scatter plots
def create_scatter_plots(df_years, indicators, countries):
    for country in countries:
        for i in range(len(indicators)):
            for j in range(i+1, len(indicators)):
                x = df_years.loc[(country, indicators[i])]
                y = df_years.loc[(country, indicators[j])]
                plt.scatter(x, y)
                plt.xlabel(indicators[i])
                plt.ylabel(indicators[j])
                plt.title(country)
                plt.show()


def subset_data(df_years, countries, indicators):
    """
    Subsets the data to include only the selected countries and indicators.
    Returns the subsetted data as a new DataFrame.
    """
    df = df_years.loc[(countries, indicators), :]
    df = df.transpose()
    return df


def calculate_correlations(df):
    """
    Calculates the correlations between the indicators in the input DataFrame.
    Returns the correlation matrix as a new DataFrame.
    """
    corr = df.corr()
    return corr


def visualize_correlations(corr):
    """
    Plots the correlation matrix as a heatmap using Seaborn.
    """
    sns.heatmap(corr, cmap='coolwarm', annot=True, square=True)
    plt.title('Correlation Matrix of Indicators')
    plt.show()


def plot_electicity_production_from_coal_sources(df_years):
    country_list = ['United States', 'India', 'China', 'Japan']
    indicator = 'Electricity production from hydroelectric sources (% of total)'
    for country in country_list:
        df_subset = df_years.loc[(country, indicator), :]
        plt.plot(df_subset.index, df_subset.values, label=country)
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title('Electricity production from hydroeletric sources (% of total)')
    plt.legend()
    plt.show()


def plot_line_CO2_emissions_kt(df_years):
    country_list = ['United States', 'India', 'China', 'Japan']
    indicator = 'CO2 emissions (kt)'
    for country in country_list:
        df_subset = df_years.loc[(country, indicator), :]
        plt.plot(df_subset.index, df_subset.values, label=country)
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title('CO2 emissions (kt)')
    plt.legend()
    plt.show()


def plot_electricity_production_from_coal_sources(df_years):
    country_list = ['United States', 'India', 'China', 'Japan']
    
    coal_sources_indicator = 'Electricity production from coal sources (% of total)'
    years = [1960, 1970, 1980, 1990, 2000]
    x = np.arange(len(country_list))
    width = 0.35

    fig, ax = plt.subplots()
    for i, year in enumerate(years):
       
        coal_sources_values = []
        for country in country_list:
            
            coal_sources_values.append(
                df_years.loc[(country, coal_sources_indicator), year])
       
        rects2 = ax.bar(x + width/2 + i*width/len(years), coal_sources_values,
                        width/len(years), label=str(year)+" "+coal_sources_indicator)

    ax.set_xlabel('Country')
    ax.set_ylabel('Value')
    ax.set_title(
        'Electricity production from coal sources')
    ax.set_xticks(x)
    ax.set_xticklabels(country_list)
    ax.legend()

    fig.tight_layout()
    plt.show()

def plot_CO2_emissions_kt(df_years):
    country_list = ['United States', 'India', 'China', 'Japan']
    CO2_emission_indicator = 'CO2 emissions (kt)'
   
    years = [1960, 1970, 1980, 1990, 2000]
    x = np.arange(len(country_list))
    width = 0.35

    fig, ax = plt.subplots()
    for i, year in enumerate(years):
        CO2_emission_values = []
        CO2_emission_values = []
        for country in country_list:
            CO2_emission_values.append(
                df_years.loc[(country, CO2_emission_indicator), year])
           
        rects1 = ax.bar(x - width/2 + i*width/len(years), CO2_emission_values,
                        width/len(years), label=str(year)+" "+CO2_emission_indicator)
       

    ax.set_xlabel('Country')
    ax.set_ylabel('Value')
    ax.set_title(
        'CO2 emissions (kt)')
    ax.set_xticks(x)
    ax.set_xticklabels(country_list)
    ax.legend()

    fig.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    df_years, df_countries = read_data(
        r"C:\Users\mouli\Downloads\wbdata.csv")

    # Call plot_electicity_production_from_oil_sources to create the  plot
    plot_electicity_production_from_coal_sources(df_years)

    # Call plot_electicity_production_from_hydroeletric_sources to create the  plot
    plot_line_CO2_emissions_kt(df_years)

    # Call plot_energy_use_and_co2_emissions to create the energy use and CO2 emissions plot
    plot_electricity_production_from_coal_sources(df_years)

    # Call plot_energy_use_and_co2_emissions to create the energy use and CO2 emissions plot
    plot_CO2_emissions_kt(df_years)
 
   # select the indicators of interest
indicators = ['Electricity production from coal sources (% of total)',
              'Electricity production from hydroelectric sources (% of total)', 'CO2 emissions (kt)']

# select a few countries for analysis
countries = ['United States', 'China', 'India', 'Japan']

# calculate summary statistics
summary_stats = calculate_summary_stats(df_years, countries, indicators)

# print the summary statistics
print_summary_stats(summary_stats)

# Use the describe method to explore the data for the 'United States'
us_data = df_years.loc[('United States', slice(None)), :]
us_data_describe = us_data.describe()
print("Data for United States")
print(us_data_describe)

# Use the mean method to find the mean Electricity production from coal sources (% of total) for each country
electicity_production_from_coal_sources = df_years.loc[(
    slice(None), 'Electricity production from coal sources (% of total)'), :]
electicity_production_from_coal_sources_mean = electicity_production_from_coal_sources.mean()
print("\nMean electicity production from coal sources for each country")
print(electicity_production_from_coal_sources)

# Use the mean method to find the mean Electricity production from hydroeletric sources (% of total) for each year
electicity_production_from_hydroeletric_sources = df_years.loc[(slice(
    None), 'Electricity production from hydroelectric sources (% of total)'), :]
electicity_production_from_hydroeletric_sources_mean = electicity_production_from_hydroeletric_sources.mean()
print("\nMean electicity production from hydroeletric sources for each country")
print(electicity_production_from_hydroeletric_sources)

df = subset_data(df_years, countries, indicators)
corr = calculate_correlations(df)
visualize_correlations(corr)


# create scatter plots
create_scatter_plots(df_years, indicators, countries)
