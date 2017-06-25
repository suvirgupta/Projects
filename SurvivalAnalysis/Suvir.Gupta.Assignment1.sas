/************************Assignment 1**********************************/
/***************** Suvir Gupta ***************************************/

libname MYLIB 'P:\SAS\SASLIB';
/* Q1 */
/* Q1.a */
/* crayon_number: char, color_name: Char, hexadecimal_code:char, RGB_triplet:char, pack_size:Numeric, 
year_issued:Numeric, year_retired:Numeric */

/* Q1.b Q1.c*/
data MYLIB.Crayon;
	infile 'P:\SAS\data\Assignment1\Crayons.dat' missover;
	length popularity $8.;
	input crayon_number 1-3 color_name $ 4-31 +1 hexadecimal_code $6. +2 RGB_triplet $ 41-55 pack_size year_issued year_retired;
	if pack_size <= 16 then popularity = 'popular';
	else if pack_size <= 48 then popularity = 'uncommon';
	else if pack_size > 48 then popularity = 'rare';
	name_wrds = countw(color_name);
run;
/* Q1.d */
proc freq data = MYLIB.Crayon ;
	table popularity / out = MYLIB.col_in_categ;
	title 'Fequency of colors falling in each category';
run;  

/* Q1.e */
data MYLIB.rare_colors;
	set MYLIB.Crayon;
	where popularity = 'rare';
run;

/* Q1.f */
proc sort data = MYLIB.rare_colors(where= (name_wrds>1))out=MYLIB.sorted_crayon;
	by year_issued color_name;
run;


PROC print DATA = MYLIB.sorted_crayon(obs =5) ;
Title 'Crayon Data Analysis';
run;


/* ------------------------------------------------------------------------------------------------------------------------------------ */
/* Q2 */

/* Q2.a Q2.b Q2.c Q2.d*/
Data MYLIB.Hotel;	
	INFILE 'P:\SAS\data\Assignment1\Hotel.dat' missover ;
	INPUT room_number number_of_guests checkin_month check_in_day checkin_year checkout_month 
		checkout_day checkout_year use_of_internet $ internet_use_days room_type $ 53-68 room_rate 69-71;
	checkin_date = input(catx('/',checkin_month,check_in_day,checkin_year ),mmddyy10.);	
	checkout_date = input(catx('/',checkout_month,checkout_day,checkout_year ),mmddyy10.);
	format checkin_date WEEKDATE10.;
	format checkout_date WEEKDATE10.;
	if use_of_internet = 'YES' then 
		Subtotal_rate = (room_rate + 10*(number_of_guests-1))*(checkout_date-checkin_date) + 4.95*internet_use_days + 9.95;
	else Subtotal_rate = (room_rate + 10*(number_of_guests-1))*(checkout_date-checkin_date);
	grand_total = Subtotal_rate + .0775*Subtotal_rate;
	format grand_total dollarx10.2;
	days_of_stay = (checkout_date-checkin_date);	

	
run;	

/* Q2.e */
Proc Sort data = MYLIB.Hotel out = MYLIB.Hotel_gtotal_sort;
	by descending grand_total days_of_stay;
run;
Proc Print data= MYLIB.Hotel_gtotal_sort(obs = 5);
run;	

/* Q2.f */
proc means data = MYLIB.Hotel Mean STDDEV MIN MAX Median Mode  N;
	class use_of_internet;
	var number_of_guests;
	title 'Descriptive Statistics of (Days of stay:No of guest)';
run;

/* Q2.g */
proc freq data = MYLIB.Hotel;
	Table  room_type*checkin_date ;
	title 'Frequency Cross Tabulation (Room type: Checkin weekday)';
run;



