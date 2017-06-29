/*Project 1 - Employee Attrition - Team 2*/
/* File Import*/

libname MYLIB  '/home/sriramrajagopal0';

proc import datafile="/home/sriramrajagopal0/FermaLogis(1).csv" 
		out=MYLIB.FermaLogis dbms=csv replace;
	getnames=YES;
run;

/* Removing unwanted columns*/

DATA MYLIB.FermaLogis(DROP=','n X Over18 EmployeeCount EmployeeNumber StandardHours);
	SET MYLIB.FermaLogis;
RUN;

/*-------------------------------------------------------------------------------------------*/

/* Data Pre-Processing*/

/*Converting "NA" values in Bonus to 0"*/

data MYLIB.fermalogis;
	set MYLIB.fermalogis;
	array bonus _character_;

	do i=1 to dim(bonus);

		if bonus(i)="NA" then
			bonus(i)="0";
	end;
run;
/* Creating a variable Censored based on the Attrition*/

DATA MYLIB.FermaLogis;
	SET MYLIB.FermaLogis;

	IF Attrition='Yes' THEN
		Censored=0;
	else
		Censored=1;
/* Converting the categorical variables into Continuous*/
	IF BusinessTravel='Non-Travel' THEN
		Travel=0;
	else if BusinessTravel='Travel_Rarely' THEN
		Travel=1;
	else 
		Travel = 2;		
	IF StockOptionLevel>0 then
		stock='Yes';
	else
		stock='No';
*Creating a new total Bonus Variable*;
	TotalBonus=bonus_1+bonus_2+bonus_3+bonus_4+bonus_5+bonus_6+bonus_7+bonus_8+bonus_9+bonus_10+
     bonus_11+bonus_12+bonus_13+bonus_14+bonus_15+bonus_16+bonus_17+bonus_18+bonus_19+bonus_20+
     bonus_21+bonus_22+bonus_23+bonus_24+bonus_25+bonus_26+bonus_27+bonus_28+bonus_29+bonus_30+
     bonus_31+bonus_32+bonus_33+bonus_34+bonus_35+bonus_36+bonus_37+bonus_38+bonus_39+bonus_40;
run;

/*-----------------------------------------------------------------------------------------------*/

/*Methdology*/

/*Statistically Significant Variables through Stepwise Regression*/
title 'Stepwise Regression Employee Atrrition';
   proc logistic data=MYLIB.FermaLogis outest=mylib.betas covout;
   	class Travel Department Education 
   		EnvironmentSatisfaction Gender JobRole JobSatisfaction JobInvolvement  MaritalStatus OverTime JobLevel PerformanceRating RelationshipSatisfaction 
   		stock WorkLifeBalance ;
      model Attrition(event='YES')= Age Travel DailyRate Department DistanceFromHome Education
      		EnvironmentSatisfaction Gender HourlyRate JobInvolvement JobRole JobLevel
      		JobSatisfaction MaritalStatus MonthlyIncome MonthlyRate NumCompaniesWorked OverTime
      		PercentSalaryHike PerformanceRating RelationshipSatisfaction TotalWorkingYears
      		TrainingTimesLastYear WorkLifeBalance YearsAtCompany YearsInCurrentRole YearsSinceLastPromotion
      		YearsWithCurrManager stock Totalbonus
                   / selection=stepwise
                     slentry=0.3
                     slstay=0.35
                     details
                     lackfit;
      output out=pred p=phat lower=lcl upper=ucl
             predprob=(individual crossvalidate);
   run;
   proc print data=mylib.betas;
      title2 'Parameter Estimates and Covariance Matrix';
   run;

/*Stepwise Regression for Time Variables to find out the statistical Significance*/
Title 'Stepwise Regression Employee Atrrition';
   proc logistic data=MYLIB.FermaLogis outest=mylib.betas_continuous covout;

      model Attrition(event='YES')= Age DailyRate DistanceFromHome 
      		HourlyRate 
      		MonthlyIncome MonthlyRate NumCompaniesWorked 
      		PercentSalaryHike TotalWorkingYears
      		TrainingTimesLastYear YearsAtCompany YearsInCurrentRole YearsSinceLastPromotion
      		YearsWithCurrManager Totalbonus
                   / selection=stepwise
                     slentry=0.3
                     slstay=0.35
                     details
                     lackfit;
      output out=pred p=phat lower=lcl upper=ucl
             predprob=(individual crossvalidate);
   run;
   proc print data=mylib.betas_continuous;
      title2 'Parameter Estimates and Covariance Matrix';
   run;
   
 /*---------------------------------------------------------------------------------------------*/  

/* Data Exploration*/
    
/*Visualization using the significant Variables*/
   
ods graphics on;
/* Biased with no overtime - so has to be removed fromthe covariates even though shown in logistic model */
PROC SGPLOT DATA = MyLIB.FERMALOGIS;
VBAR OverTime / group=Attrition groupdisplay=cluster stat = percent;
TITLE "OverTime vs Attrition";
RUN;

/*Attrition based on Job Role*/
PROC SGPLOT DATA = MyLIB.FERMALOGIS;
VBAR JobRole / group=Attrition groupdisplay=cluster stat = percent;
TITLE "JobRole vs Attrition";
RUN; 

/*Health care representative , Lab Technitian , sales representative etc are of hiigher atrition then others  */
proc freq data = MyLIB.FERMALOGIS;
   tables JobRole*Attrition / chisq  plots=mosaicplot ;
   title 'JOb role vs Attrition frequency plot';
run; 

* StockOptionLevel ;

proc freq data = MyLIB.FERMALOGIS;
   tables StockOptionLevel*Attrition / chisq  plots=mosaicplot ;
   title 'StockOptionLevel vs Attrition frequency plot';
run; 


* JobLevel;

proc freq data = MyLIB.FERMALOGIS;
   tables JobLevel*Attrition / chisq  plots=mosaicplot ;
   title 'JobLevel vs Attrition frequency plot';
run; 

* Environment Satisfaction;

proc freq data = MyLIB.FERMALOGIS;
   tables EnvironmentSatisfaction*Attrition / chisq  plots=mosaicplot ;
   title 'Environment Satisfaction vs Attrition frequency plot';
run; 


* Job Involvement;

proc freq data = MyLIB.FERMALOGIS;
   tables JobInvolvement*Attrition / chisq  plots=mosaicplot ;
   title 'Job Involvement vs Attrition frequency plot';
run; 

* Travel;

proc freq data = MyLIB.FERMALOGIS;
   tables Travel*Attrition / chisq  plots=mosaicplot ;
   title 'Travel vs Attrition frequency plot';
run; 

* DistanceFromHome;

proc freq data = MyLIB.FERMALOGIS;
   tables DistanceFromHome*Attrition / chisq  plots=mosaicplot ;
   title 'Distance From Home vs Attrition frequency plot';
run; 

/*Life Test of Time Variables*/ 

ods graphics off;
/*  lifetest withrespect to years  at company */
PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0, 3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME YearsAtCompany*Censored(0);
	title "Who are Leaving the Company";
RUN;

PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0, 3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME YearsAtCompany*Censored(0);
	strata overtime;
	title Who are Leaving the Company;
RUN;

PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0,3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME YearsAtCompany*Censored(0);
	strata JobRole;
	title Who are Leaving the Company;
RUN;


/*life test with respect to  Total working years */


PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0, 3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME TotalWorkingYears*Censored(0);
	title Who are Leaving the Company wrtTotalWorkingYears ;
RUN;


PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0, 3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME TotalWorkingYears*Censored(0);
	strata overtime / Adjust = TUKEY;
	title Who are Leaving the Company wrt TotalWorkingYears;
RUN;


PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0,3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME TotalWorkingYears*Censored(0);
	strata JobRole;
	title Who are Leaving the Company wrt TotalWorkingYears;
RUN;

/* Life Test with respect to Years since Last Promotion*/
PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0, 3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME YearsSinceLastPromotion*Censored(0);
	title Who are Leaving the Company wrt Years since Last Promotion ;
RUN;

/* Years At Company*/

PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0, 3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME YearsAtCompany*Censored(0);
	title Who are Leaving the Company wrt Years At Company ;
RUN;

/*Percent Salary Hike*/
PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0, 3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME PercentSalaryHike*Censored(0);
	title Who are Leaving the Company wrt Percent Salary Hike ;
RUN;

/* Years in Current Role*/
PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0, 3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME YearsInCurrentRole*Censored(0);
	title Who are Leaving the Company wrt Years in Current Role ;
RUN;

/* Total Bonus*/
PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0, 3, 5, 10, 
		20, 30, 40) plots=(S,H);
	TIME TotalBonus*Censored(0);
	title Who are Leaving the Company wrt Total Bonus ;
RUN;

/* Distance From Home*/
PROC LIFETEST DATA=MyLIB.fermalogis method=life INTERVALS=(0, 3, 5, 10, 
		20, 30, 40) plots=(S,H) ;
	TIME DistanceFromHome*Censored(0);
	title Who are Leaving the Company wrt Distance From Home ;
RUN;

/*-------------------------------------------------------------------------------------*/

/* Developing full model we remove the unnecessary variables from the that are low in the stepwise logistic regression */
/* using the selected variables we build the lifereg model */
/*Full Lnormal Model   */

PROC LIFEREG DATA=MYLIB.fermalogis  ;
	class Travel 
   		EnvironmentSatisfaction JobRole JobSatisfaction JobInvolvement  OverTime JobLevel RelationshipSatisfaction 
   		stock WorkLifeBalance ;

	MODEL YearsAtCompany*Censored(0)=Travel DistanceFromHome 
      		EnvironmentSatisfaction JobInvolvement JobRole JobLevel
      		JobSatisfaction NumCompaniesWorked OverTime
      		RelationshipSatisfaction TotalWorkingYears
      		WorkLifeBalance YearsSinceLastPromotion
      		YearsWithCurrManager stock 	/ D=lnormal;
	probplot;
		output out=lnormalLifeReg cdf=lnormalProb;
	title 'logNormal Distribution model';
RUN;

/* Null Log normal model */

PROC LIFEREG DATA=MYLIB.fermalogis  ;
	MODEL YearsAtCompany*Censored(0)= 	/ D=lnormal;
	probplot;
		output out=lnormalLifeReg_null ;
	title 'Lnormal null Distribution model';
RUN;



DATA calculateLogRatio;
	L_null= -1642.130921;
	L_full= -941.3043243;
	L=2 * ABS(L_full - L_null);
	p_value=1 - probchi(L, 15);
	title 'Null Lnormal Model VS Full lnormal Model';
RUN;


PROC PRINT data=calculateLogRatio;
	RUN;

/* Full Exponential model  */
PROC LIFEREG DATA=MYLIB.fermalogis  ;
	class Travel 
   		EnvironmentSatisfaction JobRole JobSatisfaction JobInvolvement  OverTime JobLevel RelationshipSatisfaction 
   		stock WorkLifeBalance ;

	MODEL YearsAtCompany*Censored(0)=Travel DistanceFromHome 
      		EnvironmentSatisfaction JobInvolvement JobRole JobLevel
      		JobSatisfaction NumCompaniesWorked OverTime
      		RelationshipSatisfaction TotalWorkingYears
      		WorkLifeBalance YearsSinceLastPromotion
      		YearsWithCurrManager stock 	/ D=Exponential;
	probplot;
		output out=ExponentialLifeReg cdf=ExponentialProb;
	title 'Exponential Distribution model';
RUN;


/* null exponential model */

PROC LIFEREG DATA=MYLIB.fermalogis  ;
	MODEL YearsAtCompany*Censored(0)= 	/ D=exponential;
	probplot;
		output out=ExponentialLifeReg_null ;
	title 'Exponential null Distribution model';
RUN;



DATA calculateLogRatio;
	L_null=-1737.498097;
	L_full= -1489.985557;
	L=2 * ABS(L_full - L_null);
	p_value=1 - probchi(L, 15);
	title 'Null Exponential Model VS Full Exponential Model';
RUN;

PROC PRINT data=calculateLogRatio;
	RUN;

/* Full Weibull Model */


PROC LIFEREG DATA=MYLIB.fermalogis  ;
	class Travel 
   		EnvironmentSatisfaction JobRole JobSatisfaction JobInvolvement  OverTime JobLevel RelationshipSatisfaction 
   		stock WorkLifeBalance ;

	MODEL YearsAtCompany*Censored(0)=Travel DistanceFromHome 
      		EnvironmentSatisfaction JobInvolvement JobRole JobLevel
      		JobSatisfaction NumCompaniesWorked OverTime
      		RelationshipSatisfaction TotalWorkingYears
      		WorkLifeBalance YearsSinceLastPromotion
      		YearsWithCurrManager stock 	/ D=Weibull;
	probplot;
		output out=WeibullLifeReg cdf=WeibullProb;
	title 'Weibull Distribution model';
RUN;


/* Null Weibull Model  */

PROC LIFEREG DATA=MYLIB.fermalogis  ;
	MODEL YearsAtCompany*Censored(0)= 	/ D=weibull;
	probplot;
		output out=WeibullLifeReg_null ;
	title 'Weibull null Distribution model';
RUN;

DATA calculateLogRatio;
	L_null=-1656.539773;
	L_full=-919.5118298;
	L=2 * ABS(L_full - L_null);
	p_value=1 - probchi(L, 15);
	title 'Null Weibull Model VS Full Weibull Model';
RUN;

/* Full Log Logistic model */
PROC LIFEREG DATA=MYLIB.fermalogis  ;
	class Travel 
   		EnvironmentSatisfaction JobRole JobSatisfaction JobInvolvement  OverTime JobLevel RelationshipSatisfaction 
   		stock WorkLifeBalance ;

	MODEL YearsAtCompany*Censored(0)=Travel DistanceFromHome 
      		EnvironmentSatisfaction JobInvolvement JobRole JobLevel
      		JobSatisfaction NumCompaniesWorked OverTime
      		RelationshipSatisfaction TotalWorkingYears
      		WorkLifeBalance YearsSinceLastPromotion
      		YearsWithCurrManager stock 	/ D=llogistic;
	probplot;
		output out=llogisticLifeReg cdf=llogisticProb;
	title 'llogistic Distribution model';
RUN;

/* Null Log Logistic model */

PROC LIFEREG DATA=MYLIB.fermalogis  ;
	MODEL YearsAtCompany*Censored(0)= 	/ D=llogistic;
	probplot;
		output out=LlogisticLifeReg_null ;
	title 'Llogistic null Distribution model';
RUN;

DATA calculateLogRatio;
	L_null=-1651.175616;
	L_full=-929.0229488;
	L=2 * ABS(L_full - L_null);
	p_value=1 - probchi(L, 15);
	title 'Null Llogistic Model VS Full Llogistic Model';
RUN;



/* Gamma model does not converge */

*Model Comparision*;

DATA CompareModels;
	L_exponential= -1489.985557;
	L_weibull= -919.5118298;
	L_lognormal=-941.3043243;
	L_logLogistic=-929.0229488;
	LRTEW=-2*(L_exponential - L_weibull);
	title Comparing Models;
RUN;

PROC PRINT DATA=CompareModels;
	title Comparing Models;
RUN;




