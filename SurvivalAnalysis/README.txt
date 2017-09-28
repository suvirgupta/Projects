

Employee attrition is one of the major concerns in all companies. FermaLogis is a pharmaceutics company 
facing this problem and we are hired to find the reasons for this cause. As per domain knowledge, there 
could be various reasons such as working overtime, job satisfaction level, business travel,employee working 
department etc.that could lead to employee turn over. FermaLogis dataset measures these variables and reports 
the proportions of employess that have left the company. In this survival analysis, we will profile the employees
who are leaving the company and what is the maximum percentage of turn over that an attribute can cause. 
We decided to split the turn over types into different categories and tried to model by the combination of similar 
turn over types. We used different techniques in order to visualize the effect of each turn over and identified 
that retirement does not attribute to attrition and hence decided to model without considering that turn over type. 

Our team has used Coxâ€™s partial likelihood estimate method to build models with censored data(considering only voluntary,
involuntary resignation and termination). Coxâ€™s regression is used to investigate the effect of variables on the time 
specified events to happen. The regression model suggests that Business travel (frequent travel),overtime and Job Involvement
are the factors that cause increase in the Hazard rate. Similarly, other factors such as Age, Total Working Years 
decreases the Hazard rate. We further analyzed the variables that affect the turn over non-proportionally and 
identified that the variable â€œYears in Current Role to be the most significant non-proportional Hazard. 
The Schoenfeld residual analysis suggests that Age, Total Working years, Years in current role, years 
 with current manager and Total bonuses are the variables which have interaction factors 
