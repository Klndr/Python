# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:21:48 2020

@author: kalan
"""

import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection='3d')

ax.scatter([0, 1, 2], [1, 3, 5], [30, 50, 70])

ax.set_xticks([0.25, 0.75, 1.25, 1.75], minor=True)

ax.set_yticks([1.5, 2.5, 3.5, 4.5], minor=True)


ax.set_zticks([35, 45, 55, 65], minor=True)


ax.tick_params(which='major', color='g', labelcolor='r',
               width=5)
#--------------------------------------------------------
'''
Plotting with keyword strings
There are some instances where you have data in 
a format that lets you access particular variables 
with strings. For example, with numpy.recarray or pandas.
DataFrame.

Matplotlib allows you provide such 
an object with the data keyword argument. 
If provided, then you may generate plots with the strings 
corresponding to these variables.
'''


#-----------------------------
line,= plt.plot([10,20,30,40], [1,2,3,4], '-')
line.set_antialiased(False) # turn off antialiasing(zig zag)


#-----------------------------------------
lines = plt.plot([10,20,30,40], [1,2,3,4])
# use keyword args
plt.setp(lines, color='g', linewidth=2.0)
# or MATLAB style string value pairs
plt.setp(lines, 'color', 'r', 'linewidth', 2.0)
#---------------------------------------------------
plt.subplot(1,2,1)
plt.plot(range(10),antialiased=False)
plt.title("AntiAliasing Off")

plt.subplot(1,2,2)
plt.plot(range(10),antialiased=True)
plt.title("AntiAliasing On")
#----------------------------------------------
'''
Working with text
text can be used to add text in an arbitrary location, 
and xlabel, ylabel and title are used to add text in the indicated locations
'''

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)
print(x)
print(len(x))
# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=1, 
                            facecolor='g', alpha=0.75)

print(n)
print(bins)
print(len(bins))
print(patches)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

#--------------------------------------------------------------

'''Annotating text
The uses of the basic text function above place 
text at an arbitrary position on the Axes. A common use for text is to annotate some feature of the plot, and the annotate method provides helper functionality to make annotations easy. In an annotation, there are two points to consider: the location being annotated represented by the argument xy and the location of the text xytext. Both of these arguments are (x, y) tuples.
'''
import numpy as np
ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
print(t)
s = np.cos(2*np.pi*t)
plt.plot(t, s, lw=2)#line width

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='yellow', 
                             shrink=0.05),
            #arrowprops={'facecolor':'black','shrink':0.05}
             horizontalalignment='right',
             verticalalignment='top'
             )
'''
shrink : move the tip and base away from the annoatated point and text
xy:position of arrow
xytext:position of the text
arrowprops:style for arrow
'''
plt.ylim(-2, 2)#set y-limits for the axes;xlim()
plt.show()
#-------------------------------------------------------------


#The xlim() function in pyplot module of matplotlib library is used to get or set the x-limits of the current axes.

'''
left: This parameter is used to set the xlim to left.
right: This parameter is used to set the xlim to right.
'''
import matplotlib.pyplot as plt 
import numpy as np 
  
h = plt.plot(np.arange(0, 10), np.arange(0, 10)) 
plt.xlim(-5,20) 
l1 = np.array((1, 1))
print(l1) 
angle = 65
  
th1 = plt.text(l1[0], l1[1], 'Line_angle', 
               fontsize = 10, rotation = angle, 
               rotation_mode ='anchor') 
# rotation_mode ='anchor'/'default'
#order of rotation and alignment  
plt.title(" matplotlib.pyplot.xlim() Example") 
plt.show() 