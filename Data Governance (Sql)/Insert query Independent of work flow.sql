--insert commands for work flow table 

insert into Step values(1,'Regulatory creates CR request');
Insert into Step Values(2,'Regulatory assigns data to launch manager in work flow table');
insert into Step values(3,'Launch Manager checks for the related attributes in DMA and email them to product Manager');
insert into Step Values(4,'Product Manager checks the attributes to be created and assign the closing date to equipment Id');
Insert into Step Values(5,'Product manager assign data owner to create the FDA specified attributes');
insert into Step Values(6,'Data owner creates data in the type 3 equipment table');
Insert into Step values(7,'Product Manager assigns data validator to the workflow table');
Insert into Step values(8,'Data validator validates the data and marks the data as approved in workflow table');


--Insert commands for the Employee table 

insert into Employee values(1,'Suvir','Regulatory',to_date('16-07-12','DD-MM-YY'));
insert into Employee values(2,'Sahil','Launch Manager',to_date('24-07-12','DD-MM-YY'));
insert into Employee values(3,'Supreet','Product Manager',to_date('16-08-12','DD-MM-YY'));
insert into Employee values(4,'Robert','Data Owner',to_date('16-09-13','DD-MM-YY'));
insert into Employee values(5,'Jhon','Data Validator',to_date('04-07-15','DD-MM-YY'));
insert into Employee values(6,'Suszi','Udi Coordinator',to_date('16-07-12','DD-MM-YY'));
insert into Employee values(7,'Andy','Regulatory',to_date('14-07-13','DD-MM-YY'));

-- Insert queries in the Equipment table:

insert into Equipment values (1,'Heart valves',3,'For use in heart surgery');
insert into Equipment values (2,'Coronary stents',3,'Used for Coronary Heart diseases');
insert into Equipment values (3,'Pacemaker',3,'To treat Bradycardia');
insert into Equipment values (4,'Powered wheelchair',2,'To assist bedridden patients');
insert into Equipment values (5,'Elastic bandages',1,'For use in musclet sprains');
insert into Equipment values (6,'Surgical Drapes',2,'For use during surgeries');
insert into Equipment values (7,'Examination gloves',1,'For use during surgeries');
insert into Equipment values (8,'Infusion pumps',1,'For use during surgeries');
insert into Equipment values (9,'Metal-on-metal hip joint',3,'For use during hip replacement surgeries');
insert into Equipment values (10,'Dental implants',3,'For use during surgeries');


-- insert into Equipment one
insert into equipment_one 
values (4,'Type 2: Powered wheel chair: Database need not to be shared with FDA');
insert into equipment_one 
values (5,'Type 1: Elastic bandages: Database need not to be shared with FDA');
insert into equipment_one 
values (6,'Type 2: Surgical Drapes: Database need not to be shared with FDA');
insert into equipment_one 
values (7,'Type 2: Examination gloves: Database need not to be shared with FDA');
insert into equipment_one 
values (8,'Type 2: Infusion pumps: Database need not to be shared with FDA');

