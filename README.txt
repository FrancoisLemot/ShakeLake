# ShakeLake
version 1

ShakeLake is a Python code designed to explore possible rupture scenarios on a user-defined fault network. It coputes the probability of exceeding a seismic intensity threshold at lakes location. Based on provided threshold for the lakes and positive or negative evidence of earthquake the code identifies the most likely rupture.

------------

## AUTHOR

Author: François Lemot
ORCID: 0000-0002-6895-6223
Contact: francois.lemot@univ-grenoble-alpes.fr

-------------

##Requirements

    Python 3.x
    Libraries numpy, matplotlib and scipy

---------------

##How to use the code
Open the script in a python environement (ex: Spyder) and run the code section by section.
You need to adjust the model parameters directly in the script.

#INPUT sections:

Faults geometry -> provide the faults geometry as an array of points coordinates.
Lake location -> coordinates for each site

Example:
	# F1=np.array([[0,0],[1,1],[2,2]])
	# F2=np.array([[1,1],[1,2],[1,3]])
	# F3=np.array([[1,0],[2,0]])
	# Faults=[F1,F2,F3]
	# L1=[0,1]
	# L2=[2,1]
	#sites=np.array([L1,L2])

Observations to fit, a list of boolean observation for each site
Ex: obs=[True, False] if lake 1 registered the earthquake but not lake 2.

Sensitivity of the lakes
Ex: sens=[5,6.5] for intensity thresholds of 5 and 6.5 respectively for lake 1 and lake 2.

#Function Definition section:

In this section, users can:

Adjust the rupture length-magnitude scaling law (e.g., Wells and Coppersmith (1994))
Adjust the intensity prediction equation (e.g., Bindi et al. (2011))

#OUTPUT sections:
You can run either of the two sections. No imput required.

##Compute and Plot Intensities for a Specific Rupture Scenario

Specify a particular rupture scenario to:
- Compute the average intensity at lake locations
- Determine the probability of satisfying observations at different sites

or

##Find the Best Fitting Rupture Scenario

Run this section to:

- Test all segments
- Identify the rupture with the highest probability of fitting the observations

Don't forget to save the figures.

-------------

## How to define possible ruptures ?

The fault network is a list of faults.
Faults are defined as a list of points (polygonal chain), with more than two points.
Points are 2-item lists of coordinates (easting, northing).
Two consecutive points define a fault segment.
Tested rupture scenarios include all continuous segment sequences in a fault.
Ruptures cannot jump over a segment or propagate from one fault to another.

If you want to check alternative rupture scenarios, you can remove the best-fit rupture from the fault network and run the model again. 

------------

## Acknowledgments

This work was funded by the French National Research Agency (ANR) under the Tibetan Orchestra project (ANR-20-CE49-0008).