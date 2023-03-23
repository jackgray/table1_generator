''' generate table 1 from redcap data
'''
import os
git_dir = '/home/gregkronberg/GitHub/narc-data-tools'
bids_tools_dir = os.path.join(git_dir,'bids_tools')
imaging_tools_dir = os.path.join(git_dir,'imaging_tools')
project_dir = os.path.join(git_dir, 'more_movie')
import sys
# sys.path.insert(1, git_dir)
# sys.path.insert(1, project_dir)
# sys.path.insert(1, bids_tools_dir)
# sys.path.insert(1, imaging_tools_dir)
import pandas as pd
import table1
import getopt


def main(argv):
    # first argument is root directory for sample, containing redcap data and subjects list
    # second argument is session number (1 or 2), corresponding to session labels in experimenter scan notes 

    # paths/directories
    ###########################################################
    root_dir = argv[0]
    if len(argv) > 1:
        session=argv[1]
    else:
        session='1'

    # variables to include in table and renaming scheme for paper formatting
    ##################################################################
    # all variables in the order that you want them to appear in the table (see README.md for variable descriptions)
    variableslist = [
        'age',
        'sex',
        'race',
        'education',
        'verbaliq',
        'nonverbaliq',
        'bdi',
        'smoking_status',
        'ftnd',
        'cig_past30',
        'cig_num_lastuse',
        'thc_past30',
        'thc_years_reguse',
        'alc_past30',
        'alc_years_reguse',
        'smast',
        'hcq',
        'heroin_past30',
        'heroin_years_reguse',
        'sds',
        'sows',
        'heroin_abstinence',
        'medication_type',
        'methadone_dose',
        'heroin_roa'
    ]

    # variables that are specific to SUD (we don't expect answers from non-SUD subjects)
    variables_oudonly = [
        'hcq',
        'heroin_past30',
        'heroin_years_reguse',
        'sds',
        'sows',
        'heroin_abstinence',
        'medication_type',
        'methadone_dose',
        'suboxone_dose'
        'heroin_roa'
    ]

    # map variable names to description in final paper-ready table 1
    rename_map = {
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

    # load and preprocess raw redcap data
    ########################################################
    # load subject list
    subjects_path = os.path.join(root_dir, 'subjects.csv')
    subs_df = pd.read_csv(subjects_path)

    # load redcap data and labels
    data_path = os.path.join(root_dir, 'MOREProject_DATA.csv')
    labels_path = os.path.join(root_dir, 'MOREProject_DATA_LABELS.csv')
    raw_data = pd.read_csv(data_path, low_memory=False)
    raw_labels = pd.read_csv(labels_path, encoding='iso-8859-1', low_memory=False)

    # preprocess raw redcap table
    df = table1.MORE.preprocess_raw_redcap_export(df=raw_data, subjects=subs_df['subject'])

    # get subject level variables from redcap table
    ########################################################
    # load table1 generator
    tabler = table1.MORE(root_dir=root_dir, df_redcap=df, df_labels=raw_labels, session=session)

    # get variables in a subject level table 
    tabler.get_subject_level_variables()

    # save subject level table
    filename = 'subject_level_variables.csv'
    outpath = os.path.join(root_dir, filename)
    tabler.dft1.to_csv(outpath, index=False)

    # get column types for each variable (categorical or continuous variable)
    column_types = tabler.dft1.apply(lambda x:tabler.get_coltype(x),axis=0)

    # organize variable types
    # get variables that are actually in table and report missing
    missing = [v for v in variableslist if v not in tabler.dft1.columns]
    variableslist = [v for v in variableslist if v in tabler.dft1.columns]
    variables_oudonly = [v for v in variables_oudonly if v in tabler.dft1.columns]
    print('missing variables from redcap data: {}'.format(missing))

    # group comparison
    variables_groupcomp = [v for v in variableslist if v not in variables_oudonly]

    # get continuous and categorical variables as separate lists
    vartype = column_types.loc[variableslist]
    variables={}
    variables['cont']=vartype[vartype=='cont'].index.tolist()
    variables['cat']=vartype[vartype=='cat'].index.tolist()

    # run statistical tests and save stats table    
    ##################################################################
    # run tests on specified variables
    df_tests = tabler.run_group_comparisons(tabler.dft1, variables)

    # save tests table
    filename = 'statistical_tests.csv'
    outpath = os.path.join(root_dir, filename)
    df_tests.to_csv(outpath, index=False)

    # turn into table 1
    ##################################################################
    # reorder and rename variables in stats df
    reorder_variables = variableslist

    # reorder and rename variables
    df_tests = df_tests.set_index('variable').loc[reorder_variables].reset_index()
    df_tests['variable_clean']=df_tests['variable'].map(rename_map)
    df_tests = df_tests.rename(index=rename_map)

    # split into sud_only and group comparisons
    dfoud = df_tests.set_index('variable').loc[variables_oudonly].reset_index().set_index('group').loc['oud'].reset_index()
    dfgroupcomp = df_tests.set_index('variable').loc[variables_groupcomp].reset_index()

    # convert to presentable table 1
    tablecomp = tabler.statistical_tests_to_table(dfgroupcomp)
    tableoud = tabler.statistical_tests_to_table(dfoud)
    tableall = pd.concat([tablecomp, tableoud], axis=0)

    # reorder variables in final table (order seems to be destroyed by groupby method above)
    cleaned_vars = []
    for v in variableslist:
        if v in rename_map.keys() and rename_map[v] in tableall.index.values:
            cleaned_vars.append(rename_map[v])
    tableall = tableall.loc[cleaned_vars]

    # save table 1 in presentable format
    filename = 'table1_formatted.csv'
    outpath = os.path.join(root_dir, filename)
    tableall.to_csv(outpath)

    # generate report for special cases
    filename = 'report_special_cases.xlsx'
    outpath = os.path.join(root_dir, filename)
    tabler.generate_report(outpath)


if __name__=='__main__':
    # first argument is root directory for sample, containing redcap data and subjects list
    # second argument is session number (1 or 2), corresponding to session labels in experimenter scan notes 
    main(sys.argv[1:])