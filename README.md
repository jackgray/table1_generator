# Generate Table 1 from redcap data
## create a directory for your sample and save raw data
    * name your sample and create a directory with that name to work in
    
    * the generate_table1.py script in this repository will expect the following files in your sample directory:
        
        1. MOREProject_DATA.csv and MOREProject_LABELS.csv (required)
            * these are the raw data and labels downloaded from redcap. simply export the full dataset for the project from redcap using default settings 
        
        2. MORE_MRI_Session_Notes.tsv (required)
            * experimenter scan notes from google drive downloaded as a tsv file (make sure it is TSV, not CSV)
            * link for MORE: https://docs.google.com/spreadsheets/d/1aj2zDIvikxN_r_PcFxAbs-ZQQR1P1BGYZtBMIVxBgK0/edit#gid=0

        3. subjects.csv (required)
            * this is your subject list
            * there should be at least two columns: "subject" and "group"
            * subject values should correspond to NARC ID's in the format S22001
        
        4. manual_data.tsv (optional)
            * you can use this file to manually overwrite any subject-level data in your table 1
            * there should be the following columns: [subject, session, variable	value, 	notes, 	experimenter, 	date]
            * the variable column should be the name of the variable you want to overwrite (see below for current variable options)

## run generate_table1.py
    1. open a terminal and navigate to this code directory
    2. run the following command: python generate_table1.py [full path to sample directory] [session number]
        * as of now, session number should be 1 or 2 corresponding to first or second MRI session (this maps to the session names in MORE_MRI_Session_Notes.tsv)

## outputs
    * table1_formatted.csv
        * this is the final table 1, formatted to be copied to a paper.  It will take a little manually cleaning up, but it should be pretty close to ready to go.
    * subject_level_variables.csv
        * this contains the raw variables for each subject, where each subject is a row and each variable is a column.  table1_formatted.csv is created after running statistical tests on these variables
    * statistical_tests.csv
        * this contains the results of the statistical tests run on the variables in subject_level_variables.csv
    * report_special_cases.xlsx
        * excel spreadsheet reporting on subjects with either missing or abnormal data
        * this can be used to inform the manual_data.tsv file. e.g. you can run generate_table1.py, check the report_special_cases.xlsx file, and then add any missing or abnormal data to manual_data.tsv and re-run generate_table1.py
    

## currently supported variables for MORE
    * rename_map = {
            'age':'Age',
            'sex':'Sex',
            'race':'Race',
            'education':'Education (years)',
            'verbaliq':'Verbal IQ',
            'nonverbaliq':'Nonverbal IQ',
            'bdi':'Depression (BDI)',
            'smoking_status':'Smoking Status',
            'ftnd':'Nicotine dependence (FTND)',
            'cig_past30':'Cigarettes (days/past 30)',
            'cig_num_lastuse':'Cigarettes (number at last use)',
            'thc_past30':'THC (days/past 30)',
            'thc_years_reguse':'THC (years of regular use)',
            'alc_past30':'Alcohol (days/past 30)',
            'alc_years_reguse':'Alcohol (lifetime use years)',
            'smast':'SMAST',
            'hcq':'HCQ',
            'heroin_past30':'Heroin (days/past 30)',
            'heroin_years_reguse':'Heroin (lifetime use years)',
            'sds':'Severity of dependence (SDS)',
            'sows':'Severity of withdrawal (SOWS)',
            'heroin_abstinence':'days abstinent',
            'medication_type':'MAT type',
            'methadone_dose':'Methadone dose',
            'suboxone_dose':'Suboxone dose',
            'heroin_roa':'Heroin route of admin.'
        }


## notes on how to extend to other variables
    * create a method table1.MORE.get_variable()
        * see examples of other such methods in table1.py
    * call this method as part of the table1.MORE.get_subject_level_variables()
        * again see examples of other such methods in table1.py
    * add the variable name to generate_table1.variables_list and generate_table1.rename_map
        * see generate_table1.py for the current list of variables