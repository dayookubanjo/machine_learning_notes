"""
Visualization with Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Year = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
Temp_I = [0.72,0.61,0.65,0.68,0.75,0.90,1.02,0.93,0.85,0.99,1.02]

#Line Graph
plt.plot(Year,Temp_I) #Line graph is default plot
plt.xlabel("Year")
plt.ylabel("Temperature Index")
plt.title("Global Warming",{"fontsize":12,"horizontalalignment":"center"})
plt.show()


Month = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 

Customer1 = [12,13,9,8,7,8,8,7,6,5,8,10]

Customer2 = [14,16,11,7,6,6,7,6,5,8,9,12]

plt.plot(Month,Customer1, color="red", label="Customer1", marker="o")
plt.plot(Month,Customer2, color="green", label="Customer2", marker="^")
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Building Consumption",{"fontsize":12,"horizontalalignment": "right"})
plt.legend() #Shows the labels we created in plot as legends
plt.legend(loc="upper right") 
plt.show()


plt.subplot(1,2,1)
plt.plot(Month,Customer1, color="red", label="Customer1", marker="o")
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Building Consumption of Cust 1",{"fontsize":12,"horizontalalignment": "right"})
plt.show()

plt.subplot(1,2,2)
plt.plot(Month,Customer2, color="green", label="Customer2", marker="^")
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Building Consumption of Cust2",{"fontsize":12,"horizontalalignment": "right"})
plt.show()

#Scatter Plot
plt.scatter(Month, Customer1, color= "red", label= "Customer1")
plt.scatter(Month, Customer2, color= "blue", label= "Customer2")
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Scatter Plot of Building Consumption")
plt.grid()
plt.legend(loc = "best")
plt.show()


#Histogram: Shows you the distribution of your data 

plt.hist(Customer1, bins = 10, color= "green")
plt.xlabel("Electricity Consumption")
plt.ylabel("Number of occurences")
plt.title("Histogram")
plt.show()


#Bar Chart
plt.bar(Month,Customer1, width= 0.8, color = "b", label= "Customer1")
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Bar chart of Building Consumption")
plt.legend()
plt.show()


#Multiple bar charts in one chart
bar_width = 0.4
Month_b = np.arange(12) #creates an arrange of numbers from 0 to 11
plt.bar(Month_b,Customer1, width= bar_width, color = "b", label= "Customer1")
plt.bar(Month_b+bar_width,Customer2, width= bar_width, color = "r", label= "Customer2")
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Bar chart of Building Consumption")
plt.xticks(Month_b + (bar_width)/12, (['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] ) )
#plt.xticks(Month_b , (['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] ) )
plt.legend()
plt.show()


#Boxplot: Crtical for outlier detection
# Boxplot terms: Box Median whisker Cap
plt.boxplot(Customer1, notch= True, vert= False) #Notched shape not regular box format & horizontal boxplot not vertical
plt.boxplot(Customer1, notch= False, vert= True) 


plt.boxplot([Customer1,Customer2], patch_artist = True,
            boxprops= dict(facecolor="blue", color= "blue"),
            whiskerprops= dict(color="g"),
            capprops = dict(color="r"),
            medianprops = dict(color = 'y')
            ) #For pyplot to know that our boxes are patches which allows us to change the colors of individual boxes

plt.show()



























































































































