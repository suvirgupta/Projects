-- Trigger to update change request id in the work flow table

create or replace Trigger tr_Cr_update_wf
After insert  on change_request 
for each row 
Declare 
Begin
insert into workflow (cr_Id) values(:new.Request_id);
insert into type_3_equipments_staging (Type_3_equipment_id) values(:new.type_3_Equipment_id);
End;
select * from workflow;
-- Sequence to auto increment work flow id 
drop sequence sq_workflow_id;
create sequence sq_workflow_id
start with 1
increment by 1
nocache;

-- Trigger applied on the workflow table to auto increment workflow id column based on sequence no.

create trigger tr_wf_autoinc
before insert 
on workflow 
for each row
Begin
select sq_workflow_id.nextval
into :new.workflow_id
from dual;
End;


-- Trigger to automatically update wf status in the change request table

create trigger tr_CR_Status_approved
after update of wf_status  on workflow
for each row
when (new.wf_status='Approved' or new.wf_status='Rejected')
begin
update change_request set cr_status=:new.wf_status where Request_Id=:old.CR_ID;
end;

-- Trigger to automatically update CR status in type 3 equipment table staging
create trigger tr_Type_3_Status_approved
after update of cr_status  on change_request
for each row
when (new.cr_status='Approved' or new.cr_status='Rejected')
begin
if(:new.cr_status='Approved') then
update type_3_equipments_staging set FINAL_STATUS=:new.cr_status where type_3_equipment_id=:old.type_3_equipment_id;
insert into type_3_equipments_final columns(type_3_equipment_id, issuing_agency, device_identifier, device_count, brand_name,model_number, packaging_type,closing_date) 
select type_3_equipment_id, issuing_agency, device_identifier, device_count, brand_name,model_number, packaging_type,closing_date
from type_3_equipments_staging where type_3_equipment_id=:old.type_3_equipment_id;
end if;
if(:new.cr_status='Rejected') then
delete from type_3_equipments_staging where type_3_equipment_id=:old.type_3_equipment_id;
end if;
end;
