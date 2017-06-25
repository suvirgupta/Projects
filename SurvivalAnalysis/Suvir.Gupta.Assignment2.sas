/*********************Assignment 2 *******************************/
/*********************Suvir Gupta  *******************************/ 

Libname MYLIB 'P:\SAS\SASLIB';
/* Q1: Merge Data set */
/* Q1.a */
proc print data = MYLIB.friends; /*FRIENDS is sorted  on ID */
run;

proc print data = MYLIB.NEWINFO; /*NEWINFO is not sorted   */
run;

proc contents data = MYLIB.friends;
run;
/*Friends  */
/* number of observation = 72 */
/*  number of variable = 7 */
/* Unique Variables: no unique variable */


proc contents data = MYLIB.NEWINFO;
run;                  
/*NEWINFO  */
/* number of observation = 113 */
/*  number of variable = 9 */
/* Unique Variables: Campain donation */


/* Q1.b  */
Data MYLIB.old_friends;
	set MYLIB.FRIENDS;
run;

proc sort data = MYLIB.NEWINFO;
	by ID;
run;

Data MYLIB.FRIENDS;
	update MYLIB.OLD_FRIENDS  MYLIB.NEWINFO;
	by ID;
run;

data MYLIB.FRIENDS;
	set MYLIB.FRIENDS (drop= Donation Campaign);
run;

/* Q1.c */
ODS TRACE on;
PROC MEANS data= MYLIB.NEWINFO sum;
	class ID;
	var Donation;
	ODS output MEANS.SUMMARY = DONATION;
run;

data MYLIB.FRIENDS_donate;
	merge MYLIB.FRIENDS  DONATION ;
	by ID;
	KEEP ID LASTNAME FIRSTNAME donation_sum;
run;


/*Q2: Visualize  data  */
/* Q2.a */
proc sort data = mylib.earthquakes out=MYLIB.earthquakes;
	by year ;
run;

proc means data = mylib.earthquakes mean;
	class year; 
	var magnitude;
	ods output MEANS.SUMMARY = mean_mag;
run; 

data earthquakes;
	merge MYLIB.earthquakes mean_mag;
	by year;
run;

/* Q2.a : Scatter, Q2.b : Series, Q2.c : legend, Q2.d : refline, Q2.e */
PROC sgplot data = earthquakes (where=(year>=2000));
	scatter X = year Y = magnitude;
	series X = year Y = magnitude_mean / lineattrs=(color = red)legendlabel='Mean(magnitude)';
	Xaxis label = 'Year of earthQuake' values=(2000 to 2011 by 1);
	Yaxis label = 'Magnitude of EarthQuake';
	keylegend/NOBORDER position=bottomright;
	refline 4.0,5.0,6.0,7.0,8.0 / axis=y label= ('light' 'moderate' 'strong' 'major' 'great') 
	transparency= 0.5 lineattrs=(pattern =dash);
run;


/* Q3: Proc USe  */
/* Q3.a : Average study time of the section 2 is less compared to section 1 */ 
proc sgplot data= mylib.study_gpa;
	vbox avetime / category=section;
	title 'average time studied between two sections';
run; 
/* Q3.b */
proc sgplot data= mylib.study_gpa;
	reg y =GPA  x=avetime / nolegcli nolegclm nolegfit;
run;

/* Q3.c , Q3.d */
proc sgplot data= mylib.study_gpa;
	reg y =GPA  x=avetime /group=section  CLM = ".95" clmtransparency= 0.5 ;
	keylegend / location=outside position=right;
run;
/* Q3.e : Comment regression line seems close to mean hence overall GPA is not affected by increase in study time */
/* Comment:  but section wize there is some linear decreasing trend in GPA with increase in study hrs in section2  */
/* Comment:  but section wize there is some linear increase trend in GPA with increase in study hrs in section1  */

/* Q4 */
/* Q4.a */
proc means data = mylib.bus mean median stddev maxdec=1;
	var plan1 plan2 plan3 ;
run;

/* Q4.b */
proc sort data = MYLIB.bus;
	by day;
run;

proc tRanspose data = MYLIB.bus out= trans_bus;
	var plan1 plan2 plan3;
	by day;
run;

proc sort data = trans_bus ;
	by day _name_;
proc transpose data = trans_bus out = trans_bus2
	(rename= ( col1 = time));
	var col1 col2 col3 col4 col5;
	by day _name_;
run; 

/* Q4.b Q4.c */
proc anova data= trans_bus2 ;
	class _name_;
	model time = _name_;
	means _name_ / tukey cldiff alpha=0.05; /* tukey pairwise comparison*/
run;

/* Q4.d */
/* ANOVA test suggest that there is differnce in mean values of the differnt plans */
/* Pair wise comparison test states that mean time of plan1 is significantly different from plan2 and plan3 */
/* where as plan2 and plan3 are not significantly different */
/* And from box plot in anova its visible that plan1 has the lowest mean time */
/* Hence city should follow plan1 */





