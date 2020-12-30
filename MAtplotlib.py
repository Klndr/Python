# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

#Basic Graph
m = [0,2,4,6,8]

# Resize your Graph (dpi specifies pixels per inch. When saving probably should use 300 if possible)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(m)
# Line 1

# Keyword Argument Notation
plt.plot(m, color='blue', linewidth=2, 
         marker='.', linestyle='--', markersize=10, 
         markeredgecolor='red')

# Shorthand notation
# fmt = '[color][marker][line]'
plt.plot(m, 'r^--')
#------------------------------------------------------

y = [1, 4, 9, 16, 25,36,49, 64]
x = [1, 16, 30, 42,55, 68, 77,88]
fig = plt.figure()
plt.plot(x,y,'rH-', linewidth=2,markeredgecolor='g') # solid line with yellow colour and square marker

plt.legend(labels ='tv', loc = 'lower right') # legend placed at lower right
plt.title("Advertisement effect on sales")
plt.xlabel('medium')
plt.ylabel('sales')

plt.savefig("fig3.png",dpi=300)#should be before plt.show; else blank pic
plt.show()
#---------------------------------------------

'''
loc
right,lower right,upper right
left,lower left,upper left
center,lower center,upper center'''

y1 = [1,4, 9, 16, 25, 36, 49, 64]
y2 = [2,6, 8, 10, 12, 24, 32, 73]
x1 = [1,16, 30, 42,55, 68, 77,88]
x2 = [1,6,12,18,28, 40, 52, 65]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
l1 = ax.plot(x1,y1,'rH-') # solid line with yellow colour and square marker
l2 = ax.plot(x2,y2,'go--') # dash line with green colour and circle marker
ax.legend(labels = ('tv','smartphone'), loc = 'lower right') # legend placed at lower right
ax.set_title("Advertisement effect on sales")
ax.set_xlabel('medium')
ax.set_ylabel('sales')
plt.show()

'''
Color codes
Character	Color
‘b’	Blue
‘g’	Green
‘r’	Red
‘b’	Blue
‘c’	Cyan
‘m’	Magenta
‘y’	Yellow
‘k’	Black
‘b’	Blue
‘w’	White  
    
    
Marker codes
Character	Description
‘.’	Point marker
‘o’	Circle marker
‘x’	X marker
‘D’	Diamond marker
‘H’	Hexagon marker
‘s’	Square marker
‘+’	Plus marker
'^' Triangle marker

Line styles
Character	Description
‘-‘	Solid line
‘—‘	Dashed line
‘-.’	Dash-dot line
‘:’	Dotted line
‘H’	Hexagon marker

'''


import matplotlib.pyplot as plt
# plot a line, implicitly creating a subplot(111)
plt.plot([1,2,3])

# now create a subplot which represents the top plot of a grid with 2 rows and 1 column.
#Since this subplot will overlap the first, the plot (and its axes) previously created, will be removed
plt.subplot(211)#(row,column,cell)
plt.plot(range(12))
plt.subplot(212, facecolor='y') # creates 2nd subplot with yellow background
plt.plot(range(12))



import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot([1,2,3])
ax2 = fig.add_subplot(221, facecolor='y')
ax2.plot([1,2,3])



import numpy as np

# select interval we want to plot points at
x = np.arange(0,4.5,0.5)
            #0,0.5,1,1.5,2,2.5,3,3.5,4
            
print(x)
# Plot part of the graph as line
plt.plot(x[:6], x[:6]**2, 'r^-')#0,0.5.1.1.5,2.2.5
# Plot remainder of graph as a dot
plt.plot(x[5:], x[5:]**2, 'g^--')#2.5,3.0,3.5,4.0

# Add a title (specify font parameters with fontdict)
plt.title('Our First Graph!', 
          fontdict={'name': 'Comic Sans MS', 
                    'size': 30,
                    'color':'red',
                    'weight':'bold',
                    })

# X and Y labels
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# X, Y axis Tickmarks (scale of your graph)
plt.xticks([0,1,2,3,4,5])
plt.yticks([0,2,4,6,8,10,12,14,16])
#-----------------------------------------------------

import pandas as pd
plt.figure(figsize=[8,5])

plt.title('Gas Prices over Time (in USD)', 
          fontdict={'weight':'bold', 'size': 18})

gas = pd.read_csv("gas_prices.csv")

plt.plot(gas.Year, gas.USA, 'b.-', label='United States')
plt.plot(gas.Year, gas.Canada, 'r.-',label='Canada')
plt.plot(gas.Year, gas['South Korea'], 'g.-')
plt.plot(gas.Year, gas.Australia, 'y.-')
plt.legend()
plt.show()


# Another Way to plot many values
countries_to_look_at = ['Australia', 'USA', 'Canada', 'South Korea']
for country in gas:
    if country in countries_to_look_at:
        plt.plot(gas.Year, gas[country], marker='.',label=country)

plt.xticks(gas.Year[::3].tolist()+[2011])

plt.xlabel('Year')
plt.ylabel('US Dollars')

plt.legend()

plt.savefig('Gasprice.png', dpi=300)
plt.show()

#plotting all countries

for country in gas:
    if country!= 'Year':
        plt.plot(gas.Year, gas[country], marker='.',label=country)

plt.xticks(gas.Year[::3].tolist()+[2011])

plt.xlabel('Year')
plt.ylabel('US Dollars')

plt.legend()

plt.savefig('Gasprice.png', dpi=300)
plt.show()
#---------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(1, 2, figsize=(10,4))
x = np.arange(1,5)
axes[0].plot( x, np.exp(x))
axes[0].plot(x,x**2)
axes[0].set_title("Normal scale")
axes[1].plot (x, np.exp(x))
axes[1].plot(x, x**2)
axes[1].set_yscale("log")
axes[1].set_title("Logarithmic scale (y)")
axes[0].set_xlabel("x axis")
axes[0].set_ylabel("y axis")
axes[0].xaxis.labelpad = 10
axes[1].set_xlabel("x axis")
axes[1].set_ylabel("y axis")
plt.show()
#-----------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import math
#plotting sine and cosine wave
x = np.arange(0, math.pi*2, 0.05)
print(x)
fig=plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.55, 0.55, 0.3, 0.3]) # inset axes
y = np.sin(x)
axes1.plot(x, y, 'b')
axes2.plot(x,np.cos(x),'r')
axes1.set_title('sine')
axes2.set_title("cosine")
plt.show()

#--------------------------------------

#plotting values using list and range(not np.arange)
x=list(range(10))
print(x)
y=[i**2 for i in x]#create a list y
plt.plot(x,y,"g")

#---------------------------------------

import matplotlib.pyplot as plt

fig,a =  plt.subplots(2,2)
import numpy as np
x = np.arange(1,5)
a[0][0].plot(x,x*x)
a[0][0].set_title('square')
a[0][1].plot(x,np.sqrt(x))
a[0][1].set_title('square root')
a[1][0].plot(x,np.exp(x))
a[1][0].set_title('exp')
a[1][1].plot(x,np.log10(x))
a[1][1].set_title('log')
plt.show()
fig.savefig("chart1.png")
#to save a particular axes
extent=a[0][0].get_window_extent()\
    .transformed(fig.dpi_scale_trans.inverted())
fig.savefig("a0.png",bbox_inches=extent.expanded(1,1))
#--------------------------------------------

import matplotlib.pyplot as plt
a1 = plt.subplot2grid((3,3),(0,0),colspan = 2)
a2 = plt.subplot2grid((3,3),(0,2), rowspan = 3)
a3 = plt.subplot2grid((3,3),(1,0),rowspan = 2, 
                      colspan = 2)
import numpy as np
x = np.arange(1,10)
a2.plot(x, x*x)
a2.set_title('square')
a1.plot(x, np.exp(x))
a1.set_title('exp')
a3.plot(x, np.log(x))
a3.set_title('log')
plt.tight_layout()
plt.show()

#--------------------------------------------------
#Bar graph
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,79,12]
ax.bar(langs,students)
plt.show()
#OR
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,79,12]
plt.bar(langs,students)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
data = [[30, 25, 50, 20],
        [40, 23, 51, 17],
        [35, 22, 45, 19]]
X = np.arange(4)
print(X)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
#ax.bar(x, height, width, bottom, align)
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)

#-------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
x = np.arange(N) # the x locations for the groups
width = 0.35
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x, menMeans, width, color='r')
ax.bar(x, womenMeans, width,bottom=menMeans, color='b')
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
#ax.set_xticks(x)
ax.set_xticklabels(['','G1','G2','G3','G4','G5'])
ax.set_yticks(np.arange(0, 81, 10))
ax.legend(labels=['Men', 'Women'])
plt.show()

#-----------------------------------------------------------

from matplotlib import pyplot as plt
import numpy as np
fig,ax = plt.subplots(1,1)
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
ax.hist(a, bins = [0,25,50,75,100])#bins -divides into categories
ax.set_title("histogram of result")
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('marks')
ax.set_ylabel('no. of students')
plt.show()
#-----------------------------------------------------------------
#PIE CHART
from matplotlib import pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')#draw a circle
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,39,12]
ax.pie(students, labels = langs,autopct='%1.2f%%')
plt.show()


'''
The default startangle is 0, which would start the "Frogs"
 slice on the positive x-axis. 
 startangle = 90 such that everything is rotated 
 counter-clockwise by 90 degrees, 
 and the frog slice starts on the positive y-axis.
'''
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
label = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
exp= (0, 0.2, 0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=exp, labels=label, autopct='%1.1f%%',
        shadow=True, startangle=90,radius=2)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
#-------------------------------------------------------
'''
shadowbool, default: False
Draw a shadow beneath the pie.

labelslist, default: None
A sequence of strings providing the labels for each wedge

startanglefloat, default: 0 degrees
The angle by which the start of the pie is rotated, counterclockwise from the x-axis.

radiusfloat, default: 1
The radius of the pie.

'''
import matplotlib.pyplot as plt

# Some data
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
fracs = [15, 30, 45, 10]

# Make figure and axes
fig, axs = plt.subplots(2, 2)
# A standard pie plot
axs[0, 0].pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True)
# Shift the second slice using explode
axs[0, 1].pie(fracs, labels=labels, autopct='%.0f%%', shadow=True,
              explode=(0, 0.1, 0, 0))
# Adapt radius and text size for a smaller pie
patches, texts, autotexts = axs[1, 0].pie(fracs, labels=labels,
                                          autopct='%.0f%%',
                                          textprops={'size': 'smaller'},
                                          shadow=True, radius=0.5)
# Make percent texts even smaller
plt.setp(autotexts, size='x-small')
autotexts[0].set_color('white')
# Use a smaller explode and turn of the shadow for better visibility
patches, texts, autotexts = axs[1, 1].pie(fracs, labels=labels,
                                          autopct='%.0f%%',
                                          textprops={'size': 'smaller'},
                                          shadow=False, radius=0.5,
                                          explode=(0, 0.05, 0, 0))

print(patches)#wedge object
print(texts)#labels
print(autotexts)#labels
plt.setp(autotexts, size='x-small')
autotexts[0].set_color('white')

plt.show()
#-----------------------------------------------------------------

import matplotlib.pyplot as plt
girls_grades = [89, 90, 70, 89, 90, 80, 90, 100, 80, 34]
boys_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]
grades_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(grades_range, girls_grades, color='r')
ax.scatter(grades_range, boys_grades, color='b')
ax.set_xlabel('Grades Range')
ax.set_ylabel('Grades Scored')
ax.set_title('scatter plot')
plt.show()


#------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-3.0, 3.0, 100)
print(xlist)
ylist = np.linspace(-3.0, 3.0, 100)
#linspace-creates a evenly spaced sequence in specified interval
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X**2 + Y**2)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()


#---------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
x,y = np.meshgrid(np.arange(-2, 2, .2), 
                  np.arange(-2, 2, .25))
z = x*np.exp(-x**2 - y**2)
v, u = np.gradient(z, .2, .2)
fig, ax = plt.subplots()
q = ax.quiver(x,y,u,v)
plt.show()

#------------------------------------------------------------