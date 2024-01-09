#importing all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading csv file
df=pd.read_csv("C:/Users/UMA/Documents/Sariya/Annual Greenhouse Gas (GHG) Air Emissions Accounts.csv")
df
df.columns
df["Country"].value_counts()#unique names of countries and count
df['Industry'].value_counts() #unique industry names and count
df['Gas_Type'].value_counts() #unique gas type of names and count
df.isnull().sum() #checking null values count 
df.info() #showing the information of dataset
df = df.drop('ISO2', axis=1) #drop the null column
df

# Filter the data based on 'Gas_Type' being 'carbondioxide'
def plot_percentage_of_co2_by_industry(df):
    
    #Filtering data for carbon dioxide emissions
    co2_data = df[df['Gas_Type'] == 'Carbon dioxide']

    #Group by 'Industry' and calculate the sum of carbon dioxide emissions for each industry
    industry_co2_sum = co2_data.groupby('Industry')[['F2010', 'F2011', 'F2012', 'F2013', 'F2014',
                                                     'F2015', 'F2016', 'F2017', 'F2018', 'F2019',
                                                     'F2020', 'F2021', 'F2022']].sum()

    # Calculating the total carbon dioxide emissions across all years for each industry
    industry_total_co2 = industry_co2_sum.sum(axis=1)

    # Calculating the percentage of total emissions for each industry
    industry_percentage_total = (industry_total_co2 / industry_total_co2.sum()) * 100

    # Displaying the result
    print(industry_percentage_total)

    # Plotting the pie chart
    values = industry_percentage_total.values
    labels = industry_percentage_total.index
    explode = [0.1] * len(labels)
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightblue']

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(values, labels=None, explode=explode, shadow=True,
                                       autopct='%1.1f%%', startangle=140,
                                       wedgeprops=dict(width=0.4, edgecolor='w'), colors=colors)

    # Adding legend with industry names
    plt.legend(labels, title="Industry Types", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.title('Percentage of Total Carbon Dioxide Emissions by Industry')
    plt.axis('equal')

    # Displaying percentages outside the pie chart with lines
    for w, p, label in zip(wedges, values, labels):
        ang = (w.theta2 - w.theta1) / 2. + w.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        plt.annotate(f'{p:.1f}%', (x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                     horizontalalignment=horizontalalignment, fontsize=10,
                     arrowprops=dict(arrowstyle="->", lw=0.5, connectionstyle=connectionstyle))

    # Displaying percentages inside boxes
    for autotext, label in zip(autotexts, labels):
        autotext.set_bbox(dict(boxstyle='round,pad=0.3', edgecolor='w', facecolor='w'))




def plot_total_co2_trend_by_country(df):
    
    # Filtering data for carbon dioxide emissions
    co2_data = df[df['Gas_Type'] == 'Carbon dioxide']

    # Group by 'Country' and calculate the sum of carbon dioxide emissions for each country
    country_co2_sum = co2_data.groupby('Country')[['F2010', 'F2011', 'F2012', 'F2013', 'F2014',
                                                   'F2015', 'F2016', 'F2017', 'F2018', 'F2019',
                                                   'F2020', 'F2021', 'F2022']].sum()

    # Calculating the total carbon dioxide emissions across all years for each country
    country_total_co2 = country_co2_sum.sum(axis=1)

    # Creation of  an area line chart with grids and pink background
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.gca().set_facecolor('pink')  # Setting background color to pink

    # Plot lines with filled area underneath
    for country in country_co2_sum.index:
        plt.fill_between(country_co2_sum.columns, country_co2_sum.loc[country], label=country, alpha=0.3)

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Total CO2 Emissions (Million Metric Tons)')
    plt.title('Total CO2 Emissions Trend by Country')

    # Adding grid lines
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adding legend
    plt.legend(title="Country", loc="upper left", bbox_to_anchor=(1, 1))

    



def plot_percentage_by_gas_type_over_years(df):
    
    # Group by 'Gas_Type' and calculating the sum of emissions for each gas type
    gas_type_sum = df.groupby('Gas_Type')[['F2010', 'F2011', 'F2012', 'F2013', 'F2014',
                                           'F2015', 'F2016', 'F2017', 'F2018', 'F2019',
                                           'F2020', 'F2021', 'F2022']].sum()

    # Calculating the total emissions across all years for each gas type
    total_gas_type = gas_type_sum.sum(axis=1)

    # Calculating the percentage of total emissions for each gas type
    percentage_gas_type = (gas_type_sum.T / total_gas_type * 100).T

    # Creation of a horizontal stacked bar chart without partition lines
    plt.figure(figsize=(12, 8))
    sns.set_palette("husl")
    sns.set_style("whitegrid")

    percentage_gas_type.plot(kind='barh', stacked=True, edgecolor='none')

    # Adding labels and title
    plt.ylabel('Gas Type')
    plt.xlabel('Percentage of Total Emissions')
    plt.title('Percentage of Emissions by Gas Type Over the Years')

    # Adding legend
    plt.legend(title="Year", loc="upper left", bbox_to_anchor=(1, 1))

    # Removing grid lines
    plt.grid(False)

    




def plot_percentage_by_gas_type_for_each_country(df):
    
    data = df[['Country', 'Gas_Type', 'F2010', 'F2011', 'F2012', 'F2013', 'F2014',
               'F2015', 'F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021', 'F2022']]

    # Group by 'Country' and 'Gas_Type' and calculate the sum of emissions
    grouped_data = data.groupby(['Country', 'Gas_Type']).sum()

    # Calculating the total emissions across all years for each country
    total_country = grouped_data.groupby('Country').sum()

    # Calculating the percentage of total emissions for each gas in each country
    percentage_country_gas = (grouped_data.T / total_country.T * 100).T.reset_index()
    melted_data = pd.melt(percentage_country_gas, id_vars=['Country', 'Gas_Type'],
                          value_vars=['F2010', 'F2011', 'F2012', 'F2013', 'F2014',
                                      'F2015', 'F2016', 'F2017', 'F2018', 'F2019',
                                      'F2020', 'F2021', 'F2022'],
                          var_name='Year', value_name='Percentage')

    # bandwidth (width of the bars) and angle of the x-axis labels
    bandwidth = 0.4
    angle = 45  # Change the angle of x direction of text

    # remove both spines and grid lines
    sns.set_style("white")

    plt.figure(figsize=(14, 8))

    # Creating a bar plot
    sns.barplot(x='Country', y='Percentage', hue='Gas_Type', data=melted_data, dodge=True,
                palette='Set1', saturation=0.7, errorbar=None, errwidth=1.5, capsize=0.15, linewidth=1.5)

    # parameters of plot
    plt.title('Percentage of Emissions by Gas Type for Each Country')
    plt.xlabel('Country')
    plt.ylabel('Percentage of Total Emissions')
    plt.xticks(rotation=angle, ha='right')
    plt.legend(title='Gas Type')

plot_percentage_of_co2_by_industry(df)    

plot_total_co2_trend_by_country(df)

plot_percentage_by_gas_type_over_years(df)

plot_percentage_by_gas_type_for_each_country(df)


def load_save_and_display_image(input_path, output_path, dpi=300):
    # Load the image
    img = plt.imread(input_path)
    # Save the image
    plt.savefig(output_path, dpi=dpi)

    # Display the image
    plt.imshow(img)
    plt.show()

input_path = "C:/Users/UMA/Downloads/22090314300.png"
output_path = "22090314300_saved.png"
load_save_and_display_image(input_path, output_path, dpi=300)




















