''' for wrat and wasi seems like we still need to do manual scoring
add aoption to check for manual wrat and wasi tables then add them to table 1 if they exist
'''
import sys
import os
git_dir = '/home/gregkronberg/GitHub/narc-data-tools'
bids_dir = os.path.join(git_dir,'bids_tools')
sys.path.append(git_dir)
sys.path.append(bids_dir)
import pandas as pd
import numpy as np
import gkutils 
# import metadata
import re
from scipy import stats
import dateutil


class Table1(object):
    
    def __init__(self, ):
        pass
    
    def get_numeric(self, string):
        ''' parse string for numeric and non numeric string components
        ==Args==
        -string:str: string to be parsed for numeric and non-numeric components

        ==Return==
        -numeric_values:list:instances of matching numeric values
        -nonnumeric_values:list:instances of nonnumeric strings between numeric values
        -numeric_types:list[list]:for each numeric value a sub-list of potential types of values (see numeric_type_patterns)
        '''
        numeric_patterns = [
            '[0-9]+\s*-\s*[0-9]+',
            '[0-9]+\s*to\s*[0-9]+',
            '[0-9]*\.[0-9]*',
            '[0-9]+/[0-9]+',
            '\$[0-9]+\s*-\s*\$?[0-9]+',
            '\$[0-9]+\s*to\s*\$?[0-9]+',
            '\$[0-9]+',
            '[0-9]+\$',
            '\d[0-9]*',
            ]

        numeric_type_patterns = {
            'money':'\$',
            'range':'[0-9]+\s*-\s*[0-9]+|[0-9]+\s*to\s*[0-9]+',
            'year':'(19[0-9]{2}|20[0-9]{2})(.*19[0-9]{2}|20[0-9]{2})?',
            'age_range':'((\D{1}|^)[0-9]{2}\D*-\D*[0-9]{2}(\D|$))|((\D{1}|^)[0-9]{2}\D*to\D*[0-9]{2}(\D|$))'
            }
            # '((\D{1}|^)[0-9]{2}(\D|$))'

        # compile search strings
        numeric_search_string = re.compile('|'.join(numeric_patterns))
        # get numeric values
        numeric_values = re.findall(numeric_search_string, string)
        # get non-numerics before and after each numeric
        nonnumeric_values = re.split(numeric_search_string, string)
        # # get types of values for each numeric
        # numeric_types=[]
        # # iterate numerics
        # for _i, _val in enumerate(numeric_values):
        #     numeric_types.append([])
        #     # iterate potential numeric types
        #     for _key, _pat in numeric_type_patterns.items():
        #         # if any matches
        #         if re.findall(_pat, _val):
        #             # add the type 
        #             numeric_types[_i].append(_key)

        return numeric_values, nonnumeric_values#, numeric_types 

    def get_numeric_types(self, numeric, nonnumeric=None):
        '''
        '''
        numeric_type_patterns = {
            'money':'\$',
            'range':'[0-9]+\s*-\s*[0-9]+|[0-9]+\s*to\s*[0-9]+',
            'year':'(19[0-9]{2}|20[0-9]{2})(.*19[0-9]{2}|20[0-9]{2})?',
            'age_range':'((\D{1}|^)[0-9]{2}\D*-\D*[0-9]{2}(\D|$))|((\D{1}|^)[0-9]{2}\D*to\D*[0-9]{2}(\D|$))'
            }
        nonnumeric_pre_type_patterns = {
            'age':'.*age\s*$',
            'year':'.*from\s*$'
        }

        nonnumeric_post_type_patterns = {
            'age':'(.*years\s*of\s*age.*)|(.*years\s*old)',
            'duration_years':'^\s*year(s?)\s*$',
            'days':'^\s*day(s?)\s*$',

        }
        # get types of values for each numeric
        numeric_types=[]
        # iterate numerics
        for _i, _val in enumerate(numeric):
            numeric_types.append([])
            # iterate potential numeric types
            for _key, _pat in numeric_type_patterns.items():
                # if any matches
                if re.findall(_pat, _val):
                    # add the type 
                    numeric_types[_i].append(_key)
            for _key, _pat in nonnumeric_pre_type_patterns.items():
                # if any matches
                if re.findall(_pat, nonnumeric[_i]):
                    # add the type 
                    numeric_types[_i].append(_key)
            for _key, _pat in nonnumeric_post_type_patterns.items():
                # if any matches
                if re.findall(_pat, nonnumeric[_i+1]):
                    # add the type 
                    numeric_types[_i].append(_key)

        return numeric_types

    def is_range(self, string):
        '''
        '''
        patterns = [
        '[0-9]+\s*-\s*[0-9]+',
        '[0-9]+\s*to\s*[0-9]+']

        search_string = re.compile('|'.join(patterns))
        if len(re.findall(search_string, string))>0:
            return True
        else: 
            return False

    def range_string_to_num(self, string, out_type='array'):
        '''
        '''
        indicators = ['-','to']
        indicator_search = '|'.join(indicators)
        # find string element that indicates range (either '-' or 'to')
        range_indicator =re.findall('(-|to)',string)
        # there should only be one range indicator
        if len(range_indicator)==1:
            range_indicator=range_indicator[0]
            # use indicator to get start and end ages
            start = float(string.split(range_indicator)[0].strip())
            end = float(string.split(range_indicator)[1].strip())
            output=np.array([start, end])
        else:
            output=string
        return output

    # get column types (categorical vs continuous)
    @classmethod
    def get_coltype(cls, x, minvals=4):
        try:
            data = pd.to_numeric(x)
            if len(data.dropna().unique())>minvals or '30_day' in x.name:
                coltype='cont'
            else:
                coltype='cat'
        except:
            coltype='cat'
        return coltype

    # run group comparisons for continuous variable
    #------------------------------------------------------
    @classmethod
    def group_comparison_continuous(cls, df, col, groups=['hc','oud'], subject_col='subject', group_col='group',):
        test={}
        has_groups = df[[group_col, col]].dropna()[group_col].unique()
        df=df.set_index([group_col,subject_col])
        

        # get summary: mean, std, n_total, n_missing
        #------------------------------------------------------
        dfsum=pd.DataFrame()
        dfsum['mean']=df.groupby(group_col)[col].mean()
        dfsum['std']=df.groupby(group_col)[col].std()
        dfsum['n_total']=df.groupby(group_col).apply(lambda x:x.index.get_level_values(subject_col).unique().shape[0])
        dfsum['n_complete']=df.groupby(group_col).apply(lambda x:x[col].dropna().index.get_level_values(subject_col).unique().shape[0])
        dfsum['n_missing']=dfsum['n_total']-dfsum['n_complete']
        dfsum['variable']=col

        # if there are not enough groups to run a statistical test, return summary
        #------------------------------------------------------
        if len(has_groups)!=len(groups):
            dfsum['statistic']=np.nan
            dfsum['p']=np.nan
            dfsum['test']=np.nan
            dfsum = dfsum.reset_index()
            return dfsum

        # otherwise, run statistical tests
        #------------------------------------------------------
        # check if all groups are normal
        #------------------------------------------------------
        normal=[]
        for group in groups:
            if stats.normaltest(df.loc[group][col])[1]<0.05:
                normal.append(False)
            else:
                normal.append(True)

        # if anything is not normal, run non-parametric tests
        #------------------------------------------------------
        if not all(normal):
            
            # if 2 groups, mannwhitney
            if len(groups)==2:
                test['col']=col
                test['test']='mannwhitneyu'
                args = [df.loc[g][col].dropna() for g in groups]
                result=stats.mannwhitneyu(*args)
                test['statistic']=result[0]
                test['p']=result[1]
            
            # if moree than 2 groups, kruskalwallis
            elif len(groups)>2:
                test['col']=col
                test['test']='kruskal'
                args = [df.loc[g][col].dropna() for g in groups]
                result=stats.kruskal(*args)
                test['statistic']=result[0]
                test['p']=result[1]
        
        # if normal, run parametric test
        #------------------------------------------------------
        else:

            # if 2 groups, ttest
            if len(groups)==2:
                # test['col']=col
                test['test']='ttest_ind'
                args = [df.loc[g][col].dropna() for g in groups]
                result=stats.ttest_ind(*args)
                test['statistic']=result[0]
                test['p']=result[1]
            
            # if more than 2 groups, anova
            elif len(groups)>2:
                # test['col']=col
                test['test']='f_oneway'
                args = [df.loc[g][col].dropna() for g in groups]
                result=stats.f_oneway(*args)
                test['statistic']=result[0]
                test['p']=result[1]
        dfsum['statistic']=test['statistic']
        dfsum['p']=test['p']
        dfsum['test']=test['test']
        dfsum = dfsum.reset_index()
        return dfsum

    # run group comparisons for categorical variable
    #------------------------------------------------------
    @classmethod
    def group_comparison_categorical(cls, df, col, groups=['hc','oud'], group_col='group', subject_col='subject'):
        # categorical columns
        #---------------------------------------------------------------
        test={}
        table = df.groupby(group_col)[col].value_counts().unstack(level=0).fillna(0)
        chi2_result = stats.chi2_contingency(table)
        test['variable']=col
        test['test']='chi2'
        test[group_col]=table.columns.tolist()
        test['distributions']=['//'.join(table[c].astype(str).values) for c in table.columns]
        test['statistic']=chi2_result[0]
        test['p']=chi2_result[1]
        test['categories']='//'.join([str(v) for v in table.index.tolist()])
        dfsum=pd.DataFrame(test)
        return dfsum, table

    # run tests on all variables
    #------------------------------------------------------
    @classmethod
    def run_group_comparisons(cls, df, variables):
        dfcont=pd.DataFrame()
        dfcat = pd.DataFrame()
        for vtype, varlist in variables.items():
            for var in varlist:
                if vtype in ['categorical', 'cat', 'binary','count']:
                    result, table = cls.group_comparison_categorical(df, var)
                    dfcat = dfcat.append(result)
                elif vtype in ['continuous','cont']:
                    try:
                        result = cls.group_comparison_continuous(df, var)
                    except:
                        breakpoint()
                    dfcont = dfcont.append(result)
        df = dfcont.append(dfcat)
        return df

    # convert to presentable table 1
    #------------------------------------------------------
    @classmethod
    def statistical_tests_to_table(cls, df, variable_col='variable_clean', group_col='group'):

        # convert info into a single text box
        #------------------------------------------------------
        def get_box_info(x):
            if x['mean'].isna().all():
                if not x['distributions'].isna().any():
                    text = '; '.join([x['distributions'].values[0], x['categories'].values[0]])
                    return text
            else:
                text = '{:.2f} +- {:.2f}'.format(x['mean'].values[0], x['std'].values[0])
                return text
        
        # get box info
        table = df.groupby([variable_col, group_col]).apply(get_box_info).unstack(level=0).T

        # get statistical test box for group comparisons
        #------------------------------------------------------
        def get_test_info(x, thresh=0.05):
            # breakpoint()
            if x['p'].values[0]<thresh:
                sig='*'
            else:
                sig=''
            test = x['test'].values[0]
            stat=gkutils.round_sig(x['statistic'].values[0])
            pval = gkutils.round_sig(x['p'].values[0], sig=2)
            text = '{}={}; p={}{}'.format(test,stat, pval, sig)
            return text
        ntests = table.index.unique().shape[0]
        thresh=0.05/ntests
        print('significance threshold for {} tests: {:.3f}'.format(ntests, thresh))
        table['test'] = df.groupby([variable_col]).apply(get_test_info, thresh=thresh)
        return table

    # General methods
    ################################################
    @classmethod
    def preprocess_raw_redcap_export(cls, df, subjects=[]):
        # copy new df
        df = df.copy()

        # update column names
        df = cls.update_column_names(df)

        # replace 'NI' (no information)  and 'NASK' (not asked) with np.nan
        df = df.replace('NI', np.nan)
        df = df.replace('NASK', np.nan)
        df = df.replace('[not completed]', np.nan)

        # add narc_id to all rows 
        df = gkutils.Redcap.set_narc_id(df).reset_index()

        # pull only subjects in sample
        if subjects is not None and len(subjects) > 0:
            df = df[df['narc_id'].isin(subjects)]

        # convert date columns to ordinal
        #---------------------------------
        # date
        df = gkutils.Redcap.convert_date_columns_to_ordinal(df, date_str='_date')
        # timestamp
        df = gkutils.Redcap.convert_date_columns_to_ordinal(df, date_str='_timestamp')
        # if manual date (entered by experimenter) is unavailable, use get timestamp
        df = gkutils.Redcap.match_manual_date_to_timestamp(df=df, survey_key='demographics', date_str='date', timestamp_str='timestamp',ordinal=True)

        # # add event date for redcap events (only individual instruments are assigned dates in redcap)
        # df = gkutils.Redcap.addcol_redcap_event_date(df=df, ordinal_str='_date_ordinal')

        # # get age for MRI days
        # df = cls.propagate_age_to_all_events(df, events=['pretreatment_mri_arm_1', 'posttreatment_mri_arm_1'])

        return df

    @classmethod
    def update_column_names(cls, df):
        
        col_map = {
            'surveydate': 'demographics_date',
        }
        df = df.rename(columns=col_map)
        return df

    @classmethod
    def propagate_age_to_all_events(cls, df, events=None):
        if events is None:
            events = df.redcap_event_name.unique()
        for event in events:
            print('updating subject age for event: {}'.format(event))
            df['age'] = gkutils.Redcap.adjust_age_variable_to_event_date(df=df, variable='age', src_date_col='demographics_date_ordinal', tgt_event=event,  event_date_col='tox_date_ordinal')
        return df

    def check_date_matches_timestamp(cls):
        pass

    def normalize_duration(self, input_series, new_series, direct_map,new_val_max=None, output_type='days', default_units='years'):
        '''
        ==Args==
        '''
        # print('normalizing units for: ', input_series.name)
        # list to keep track of unhandled columns
        unhandled = []
        # iterate rows
        for row_i, row in input_series.iteritems():

            numeric_values=[]
            # skip nans
            #------------
            if pd.isna(row):
                continue

            # handle manually specified rows
            #-------------------------------
            if row in direct_map.keys():
                # print(row, direct_map[row])
                # update new series
                new_series[row_i] = direct_map[row]

            else:
                # default new_val
                new_val = np.nan

                # handle zeros
                #--------------
                if row=='0' :
                    # new_val=np.nan
                    new_val=0
                else:

                    search_patterns = [
                    '[0-9]+ days',
                    '[0-9]+ day',
                    '[0-9]+ weeks',
                    '[0-9]+ week',
                    '[0-9]+ months',
                    '[0-9]+ month',
                    '[0-9]+\.[0-9]+ month',
                    '[0-9]+ years|[0-9]+\.[0-9]+ years',
                    '[0-9]+ year|[0-9]+\.[0-9]+ year',
                    '[0-9]+yrs|[0-9]+\.[0-9]+yrs',
                    '^[0-9]+$|^[0-9]+$|[0-9]+\.[0-9]+',
                    ]
                    search_string = '|'.join(search_patterns)
                    values = re.findall(search_string, str(row))
                    # print(values)
                    # if only 1 numeric value
                    if len(values)==1:
                        
                        # print(row,numeric_values)
                        post_text = row.split(values[0])[1].strip()
                        
                        has_text = len(post_text)>0

                        if 'year' in values[0] or 'yrs' in values[0]:
                            numeric = float(re.findall('[0-9]+\.[0-9]+|[0-9]+',values[0])[0])
                            days = numeric*365
                        elif 'month' in values[0]:
                            numeric = float(re.findall('[0-9]+\.[0-9]+|[0-9]+',values[0])[0])
                            days = numeric*31
                        elif 'week' in values[0]:
                            numeric = float(re.findall('[0-9]+\.[0-9]+|[0-9]+',values[0])[0])
                            days = numeric*7
                        elif 'day' in values[0]:
                            days = float(re.findall('[0-9]+\.[0-9]+|[0-9]+',values[0])[0])
                        else:
                            # if not unit specified assume years
                            # numeric = float(re.findall('[0-9]+\.[0-9]+|[0-9]+',values[0])[0])
                            numeric=self.get_numeric(values[0])[0]
                            if len(numeric)==1:
                                numeric=float(numeric[0])
                                # print(numeric)
                                if default_units=='years':
                                    days = numeric*365
                                elif default_units=='months':
                                    days=numeric*31
                                elif default_units=='weeks':
                                    days=numeric*7
                                elif default_units=='days':
                                    days=numeric
                            else:
                                days=np.nan


                        if output_type=='days':
                            new_val=days
                        elif output_type in ['years','year','yrs','yr']:
                            new_val=days/365.
                        elif output_type=='months':
                            new_val=days/31.
                        elif output_type=='weeks':
                            new_val=days/7.
                        # print('|'.join([row,str(new_val)]))
                        # if not has_text:
                        #     new_val = values[0]

                # update the new series
                new_series[row_i] = new_val
                # track unhandled rows 
                # be sure to check why these were not handled!
                if pd.isna(new_val):
                    unhandled.append((row_i, row, new_val))

        return new_series,unhandled

    
    def generate_report(self,output_path, format='excel'):
        if format in ['excel']:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                for var, reports in self.report.items():
                    df = pd.DataFrame({var:reports})
                    df.to_excel(writer, sheet_name=var)



class MORE(Table1):

    def __init__(self, root_dir, session, df_redcap, df_labels):
        Table1.__init__(self, )

        # set attributes
        self.session =session
        self.df_redcap = df_redcap
        self.df_labels = df_labels
        self.root_dir = root_dir

        # map session numbers to redcap event names for scan days
        self.session_map_scans = {
            '1': 'pretreatment_mri_arm_1',
            '2': 'posttreatment_mri_arm_1',
        }

        # map session numbers to redcap event names for task days
        self.session_map_taskday = {
            '1': 'task_day_1_arm_1',
            '2': 'task_day_2_arm_1',
            '3': '3_month_followup_arm_1',
        }
        
        # use this to create notes, e.g. missing or abnormal data
        # self.report['variable'] = [note, note, note...]
        self.report={
        }

        # load experimenter scan notes with some clean up
        self.scan_notes_path = os.path.join(self.root_dir, 'MORE_MRI_Session_Notes.tsv')
        self.scan_notes = ScanNotes(filename=self.scan_notes_path).scan_notes
        self.scan_notes['session']=self.scan_notes['session'].astype(str)

        # load manually labeled subject-level data (assumes columns ['subject','session','variable','value'])
        manual_path = os.path.join(self.root_dir, 'manual_data.tsv')
        if os.path.exists(manual_path):
            self.manual_data = pd.read_csv(manual_path, sep='\t')
        else:
            self.manual_data = pd.DataFrame(columns = ['subject','session','variable','value', 'notes'])

        # load subjects and add scan dates for current session
        self.subjects_path = os.path.join(self.root_dir, 'subjects.csv')
        self.subs_df = pd.read_csv(self.subjects_path)
        self.subs_df=self.get_scan_dates(subs_df=self.subs_df, session=self.session, task='movie')

        # initialize subject level table 1 data (table 1, with stats, will be created from this later)
        self.dft1 = pd.DataFrame()
        self.dft1['subject'] = self.subs_df['subject']
        self.dft1['date_ordinal'] = self.subs_df['date_ordinal']
        self.dft1['session'] = self.subs_df['session']
        self.dft1['group'] = self.subs_df['group']
        


        # rename variables for final table
        self.rename_map = {
            'age':'Age',
            'sex':'Sex',
            'race':'Race',
            'education':'Education (years)',
            'verbal_iq':'Verbal IQ',
            'nonverbal_iq':'Nonverbal IQ',
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
            'methadone_dose':'methadone dose',
            'suboxone_dose':'suboxone dose',
            'her_route_admin':'HER route of administration'
        }

    def get_subject_level_variables(self):
        # get subject-level variables from redcap dataframe
        #-----------------------------------------------------------------------
        # age
        self.dft1['age'] = self.get_age(df_t1=self.dft1, df_redcap=self.df_redcap)

        # sex
        self.dft1['sex'] = self.get_sex(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels, trans_strategy='birth')

        # race
        self.dft1['race'] = self.get_race(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,)
        
        # ethnicity
        self.dft1['ethnicity'] = self.get_ethnicity(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,)

        # education
        self.dft1['education'] = self.get_education(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,)

        # BDI
        self.dft1['bdi'] = self.get_bdi(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)

        # cigarette smoking status (never/past/current)
        self.dft1['smoking_status'] = self.get_smoking_status(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)

        # FTND score (nicotine dependence)
        self.dft1['ftnd'] = self.get_ftnd(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)

        # cigarettes (days/past30)
        self.dft1['cig_past30'] = self.get_cig_past30(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # cigarettes (number at last use)
        self.dft1['cig_num_lastuse'] = self.get_cig_num_lastuse(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # thc (days/past30)
        self.dft1['thc_past30'] = self.get_thc_past30(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # thc (years regular use)
        self.dft1['thc_years_reguse'] = self.get_thc_years_reguse(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # alcohol (days/past30)
        # NOTE it looks like we only collect this variable in the ASI at screening and 3 month follow up, but not MRI or task days, so this will be inaccurate for scan dates
        self.dft1['alc_past30'] = self.get_alc_past30(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # alcohol (lifetime in years)
        self.dft1['alc_years_reguse'] = self.get_alc_years_reguse(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # SMAST
        self.dft1['smast'] = self.get_smast(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # HCQ
        self.dft1['hcq'] = self.get_hcq(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # heroin (days/past30)
        self.dft1['heroin_past30'] = self.get_heroin_past30(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        self.dft1['heroin_past30'] = self.dft1['heroin_past30'].astype('float')

        # heroin (lifetime years)
        self.dft1['heroin_years_reguse'] = self.get_heroin_years_reguse(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # SDS
        self.dft1['sds'] = self.get_sds(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)

        # SOWS
        self.dft1['sows'] = self.get_sows(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)

        # heroin days abstinent
        self.dft1['heroin_abstinence'] = self.get_heroin_abstinence(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # FIXME WRAT and WASI need to be checked for accuracy against pierre's results
        ####################################################################
        # each questionnaire shoudl just have it's own class in behavioral.py
        # get verbal IQ
        self.dft1['verbaliq'] = self.get_verbaliq(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,)

        # # get nonverbal IQ
        self.dft1['nonverbaliq'] = self.get_nonverbaliq(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,)
        # ####################################################################
        
        # FIXME check for methadone doses from treatment facility (not in redcap)
        # MAT type
        self.dft1['medication_type'] = self.get_medication_type(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)

        # methadone dose
        self.dft1['methadone_dose'] = self.get_methadone_dose(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)

        # get suboxone dose
        self.dft1['suboxone_dose'] = self.get_suboxone_dose(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)
        
        # heroin ROA
        self.dft1['heroin_roa'] = self.get_heroin_roa(df_t1=self.dft1, df_redcap=self.df_redcap, df_labels=self.df_labels,session=self.session)

        # update with manually labeled subject data
        ###########################################
        if not self.manual_data.empty:
            for _i, _row in self.manual_data.iterrows():
                if (isinstance(_row['session'], float) and pd.isna(_row['session']))or str(_row['session']) == self.session:
                    _report = 'subject: {}, value manually set to: {}'.format(_row['subject'],_row['value'])
                    if 'experimenter' in _row.keys():
                        _report += ', by experimenter: {}'.format(_row['experimenter'])
                    if 'date' in _row.keys():
                        _report += ', on date: {}'.format(_row['date'])
                    self.report[_row['variable']].append(_report)
                    self.dft1.loc[self.dft1['subject']==_row['subject'],_row['variable']] = _row['value']

    # get scan dates for each subject
    ########################################
    def get_scan_dates(self, subs_df, session, task, rescan_strategy='last'):
        ''' get scan dates for scans that are included in the sample from experimenter scan notes
        '''
        dates=[]
        dates_ordinal=[]
        # iterate over subjects in sample (current session)
        for subject in subs_df['subject']:
            
            # get notes for subject/session
            row = self.scan_notes.set_index(['subject','session']).loc[(subject,session)]
            
            # FIXME set this up to handle many keys for a scan type (e.g. cue reactivity, cuereact, cue-reactivity, etc.)
            # if the task is in the rescan notes, use the rescan date
            if not pd.isna(row.notes_rescan) and task.lower() in row.notes_rescan.lower() and rescan_strategy == 'last':
                date_str = row.date_rescan
                date_ordinal = row.date_rescan_ordinal
            
            # otherwise use the original scan date
            else:
                date_str = row.date
                date_ordinal = row.date_ordinal
            
            # keep list of dates and add subs_df
            dates.append(date_str)
            dates_ordinal.append(date_ordinal)
        subs_df['date'] = dates
        subs_df['date_ordinal'] = dates_ordinal
        subs_df['session']=session
        
        return subs_df
          
    # Get subject level variables
    ########################################
    def get_age(self, df_t1, df_redcap, age_col='age', date_col='demographics_date_ordinal'):    
        '''
        '''
        print('getting age...')
        # iterate subjects in df_t1
        ages = []
        df = df_redcap.copy()
        for i, row in df_t1.iterrows():
            # get scan date
            scan_date = row.date_ordinal

            # get redcap rows that have correct date and age information
            df_tmp = df[df['narc_id']==row.subject][[date_col,age_col]].dropna()

            # FIXME handle cases where more or less than one age are returned
            # update age based on time difference between scan date and redcap date
            age_original = df_tmp[age_col].unique()[0]
            date_original = df_tmp[date_col].unique()[0]
            date_diff = (scan_date-date_original)/365.25
            age = age_original + date_diff
            ages.append(age)

        return ages

    def get_sex(self, df_t1, df_redcap, df_labels, trans_strategy='birth'):
        '''
        '''
        print('getting sex...')
        # initialize report
        self.report['sex']=[]

        # column names in data and labels df
        data_col = 'gender'
        suffix = '_label'
        labeled_col = data_col+suffix

        # get gender labels ('male', 'female', 'trans')
        df = df_redcap.copy()
        df[labeled_col] = gkutils.Redcap.data2labels(data_df=df, labels_df=df_labels, data_columns=[data_col], return_cols='data', suffix=suffix)
        
        # iterate subjects in sample
        sexes = []
        for i, row in df_t1.iterrows():
            
            # get df for the current subject
            df_tmp = df[df['narc_id']==row['subject']][labeled_col].dropna()
            
            # correctly identified one sex for the current subject 
            if len(df_tmp.unique())==1:
                sex=df_tmp.unique()[0]

                # if male or female just append to list
                if sex in ['Male','Female'] and 'Trans' not in sex:
                    sexes.append(sex)

                # adjust for transgender
                elif 'Trans' in sex:
                    if 'Male' in sex and 'Female' in sex:
                        # get birth sex and current gender
                        male_idx = sex.index('Male')
                        female_idx = sex.index('Female')
                        if male_idx<female_idx:
                            birth_sex = 'Male'
                            current_sex = 'Female'
                        else:
                            birth_sex='Female'
                            current_sex='Male'

                        # keep biological/birth sex for table 1
                        if trans_strategy in ['birth', 'original','biological']:
                            # add report
                            _report = 'subject {} identified as {}, included in table 1 as {}'.format(row.subject, sex, birth_sex)
                            self.report['sex'].append(_report)

                            # update table 1 sex
                            sexes.append(birth_sex)

                        # keep current/identified gender for table 1
                        elif trans_strategy in ['current','presented']:
                            # add report
                            _report = 'subject {} identified as {}, included in table 1 as {}'.format(row.subject, sex, current_sex)
                            
                            self.report['sex'].append(_report)

                            # update table 1 sex
                            sexes.append(current_sex)

                        # keep as separate category for table 1
                        else:
                            # add to report 
                            _report = 'subject {} identified as {}, included in table 1 as {}'.format(row.subject, sex, sex)
                            
                            self.report['sex'].append(_report)

                            # don't update sex
                            sexes.append(sex)



                    else:
                        _report = 'subject {} identified as {}, but biological sex not specified, excluded from table 1 for now'.format(row.subject, sex)
                        
                        self.report['sex'].append(_report)
                        sexes.append(np.nan)
                
                # unhandled category
                else:
                    sexes.append(np.nan)
            
            # too many sex labels for the current subject
            elif len(df_tmp.unique())>1:
                # add to report
                _report = 'multiple sex labels for subject {}, excluded from table 1 for now'.format(row.subject)
                
                self.report['sex'].append(_report)
                # add to list
                sexes.append(np.nan)    
            
            # no sex label for the current subject
            else:
                # add to report
                _report = 'no sex label found for {}, excluded from table 1 for now'.format(row.subject)
                
                self.report['sex'].append(_report)
                # add to list
                sexes.append(np.nan)

        return sexes

    def get_race(self, df_t1, df_redcap, df_labels, hispanic=True):
        '''
        '''
        print('getting race...')
        # FIXME handle subjects that only provide Hispanic as race (race=other?)
        self.report ['race']=[]
        race_col_map = {
            'race___1':'White',
            'race___2':'Black',
            'race___3':'Asian',
            'race___4':'Native Hawaiian or Other Pacific Islander',
            'race___5':'American Indian',
            'race___6':'Alaska Native',
            'race___ni':'No information',
            'race___nask':'Not asked',
        }
        race_cols = [k for k in race_col_map.keys()]

        asi_race_map = {
            'Black (not Hisp)':'Black', 
            'White (not Hisp)':'White',
            'Asian/Pacific':'Asian',
            'American Indian':'American Indian',
            'Hispanic-Cuban':'Hispanic',
            'Other Hispanic':'Hispanic', 
            'Hispanic-Puerto Rican':'Hispanic', 
            'Hispanic-Mexican':'Hispanic', 
            
        }

        # get ethnicity labels
        # --------------------------------
        ethnicity_col = 'ethnicity'
        suffix = '_label'
        label_col = ethnicity_col+suffix
        df = df_redcap.copy()
        df[label_col] = gkutils.Redcap.data2labels(data_df=df, labels_df=df_labels, data_columns=[ethnicity_col], return_cols='data', suffix=suffix)

        # get asi race labels
        #--------------------------------
        asi_col = 'asi_race'
        suffix = '_label'
        asi_label_col = asi_col+suffix
        df[asi_label_col] = gkutils.Redcap.data2labels(data_df=df, labels_df=df_labels, data_columns=[asi_col], return_cols='data', suffix=suffix)

        
        races =[]
        # iterate subjects in sample
        for i, row in df_t1.iterrows():

            # get race
            #--------------------------------
            df_race = df[df['narc_id']==row['subject']][race_cols].dropna()

            df_race_asi = df[df['narc_id']==row['subject']][asi_label_col].dropna()

            # should just be one row and one selected response
            if df_race.shape[0]==1 and df_race.iloc[0,:].sum()==1:
                race_key = [r for r in race_cols if df_race[r].values[0]==1][0]
                race=race_col_map[race_key]
                races.append(race)

            # can't find a valid race
            else:
                # check asi race variable
                if len(df_race_asi.unique())==1:
                    # race from ASI
                    race = asi_race_map[df_race_asi.unique()[0]]

                    # ASI can return Hispanic, but we don't want to include this in table 1
                    if race=='Hispanic':
                        if  hispanic:
                            _report = 'subject: {}, could not find a valid race from demographics survey.  found {} from ASI, setting race to {}'.format(row.subject, race, race)
                            race='Hispanic'
                        else:
                            _report = 'subject: {}, could not find a valid race  from demographics survey.  found {} from ASI. however {} is not a valid race, excluding this subjects race for now'.format(row.subject, race, race)
                            race=np.nan
                    else:
                        _report  = 'subject: {}, could not find a valid race  from demographics survey, but found {} from ASI'.format(row.subject, race)
                    # add to report and race list
                    
                    self.report['race'].append(_report)
                    races.append(race)
                # if no race found in asi either, return nan and handle manually
                else:
                    _report = 'subject: {}, did not find a valid race  in demographics or ASI survey, excluded from table 1 for now'.format(row.subject)
                    
                    self.report['race'].append(_report)
                    races.append(np.nan)

        return races

    def get_ethnicity(self, df_t1, df_redcap, df_labels):
        '''
        '''
        print('getting ethnicity...')
        self.report['ethnicity']=[]
        asi_eth_map = {
            'Black (not Hisp)':'Non-hispanic', 
            'White (not Hisp)':'Non-hispanic',
            'Asian/Pacific':'Non-hispanic',
            'American Indian':'Non-hispanic',
            'Hispanic-Cuban':'Hispanic',
            'Other Hispanic':'Hispanic', 
            'Hispanic-Puerto Rican':'Hispanic', 
            'Hispanic-Mexican':'Hispanic', 
            
        }

        # get ethnicity labels
        # --------------------------------
        ethnicity_col = 'ethnicity'
        suffix = '_label'
        eth_label_col = ethnicity_col+suffix
        df = df_redcap.copy()
        df[eth_label_col] = gkutils.Redcap.data2labels(data_df=df, labels_df=df_labels, data_columns=[ethnicity_col], return_cols='data', suffix=suffix)

        # get asi race labels
        #--------------------------------
        asi_col = 'asi_race'
        suffix = '_label'
        asi_label_col = asi_col+suffix
        df[asi_label_col] = gkutils.Redcap.data2labels(data_df=df, labels_df=df_labels, data_columns=[asi_col], return_cols='data', suffix=suffix)

        
        ethnicities =[]
        # iterate subjects in sample
        for i, row in df_t1.iterrows():

            # get race
            #--------------------------------
            df_eth = df[df['narc_id']==row['subject']][eth_label_col].dropna()

            df_race_asi = df[df['narc_id']==row['subject']][asi_label_col].dropna()

            # should just be one row and one selected response
            if len(df_eth.unique())==1:
                eth = df_eth.unique()[0]
                ethnicities.append(eth)

            # can't find a valid ethnicity in demographics survey
            else:
                # check asi race variable
                if len(df_race_asi.unique())==1:
                    # ethnicity from ASI
                    ethnicity = asi_eth_map[df_race_asi.unique()[0]]
                    # add to report
                    _report = 'ethnicity not found in demographics for {}, but could be inferred from ASI: {}'.format(row.subject, ethnicity)
                    
                    self.report['ethnicity'].append(_report)
                    # add to output
                    ethnicities.append(ethnicity)
                # no ethnicity found in asi either
                else:
                    # add to report
                    _report = 'ethnicity not found in demographics or ASI for {}'.format(row.subject)
                    
                    self.report['ethnicity'].append(_report)

                    # add to output
                    ethnicities.append(np.nan)

        return ethnicities
    
    def get_education(self, df_t1, df_redcap, df_labels):
        '''
        '''
        print('getting education...')
        self.report['education']=[]
        edu_col = 'yearsinschoolself'

        educations =[]
        for i, row in df_t1.iterrows():
            # get education
            #--------------------------------
            df_edu = df_redcap[df_redcap['narc_id']==row['subject']][edu_col].dropna()

            # should just be one education value
            if len(df_edu.unique())==1:
                edu = df_edu.unique()[0]
                educations.append(edu)
            
            # can't find a valid education
            else:
                # add to report
                _report = 'education not found in demographics for {}'.format(row.subject)
                
                self.report['education'].append(_report)
                # add to output
                educations.append(np.nan)
        return educations
        
    def get_verbaliq(self, df_t1, df_redcap, df_labels):
        '''
        '''
        print('getting verbaliq...')
        self.report['verbaliq']=[]
        verbaliq_col = 'wrat_total_reading'
        version_col = 'wrat_version' # 0=blue, 1=tan
        age_col = 'age'

        verbaliqs =[]
        for i, row in df_t1.iterrows():
            # get verbaliq
            #--------------------------------
            df_verbal = df_redcap[df_redcap['narc_id']==row['subject']][verbaliq_col].dropna()

            # get wrat verion
            #--------------------------------
            df_version = df_redcap[df_redcap['narc_id']==row['subject']][version_col].dropna()

            # age
            age = row[age_col]

            wrat_scaler = WRAT3()
            # should just be one verbaliq value
            if len(df_verbal.unique())==1 and len(df_version.unique())==1:
                version = df_version.unique()[0]
                if version==0:
                    version = 'blue'
                elif version==1:
                    version = 'tan'
                wrat_total = df_verbal.unique()[0]
                wrat_scaled = wrat_scaler.scale_raw_score(wrat_total=wrat_total, age=age, version=version)
                if wrat_scaled =='below_45':
                    wrat_scaled = 45
                wrat_scaled = float(wrat_scaled)
                verbaliqs.append(wrat_scaled)
            
            # can't find a valid verbaliq
            else:
                # add to report
                _report = 'subject: {}, verbaliq not found for'.format(row.subject)
                
                self.report['verbaliq'].append(_report)
                # add to output
                verbaliqs.append(np.nan)
        return verbaliqs

    def get_nonverbaliq(self, df_t1, df_redcap, df_labels):
        '''
        '''
        print('getting nonverbaliq...')
        self.report['nonverbaliq']=[]
        nonverbaliq_col = 'wasi_total'

        nonverbaliqs =[]
        for i, row in df_t1.iterrows():
            # get nonverbaliq
            #--------------------------------
            df_nonverbal = df_redcap[df_redcap['narc_id']==row['subject']][nonverbaliq_col].dropna()

            # get age
            #--------------------------------
            age = row.age

            wasi_scaler = WASI()

            # should just be one nonverbaliq value
            if len(df_nonverbal.unique())==1:
                nonverbal = df_nonverbal.unique()[0]
                wasi_scaled, wasi_tscore = wasi_scaler.scale_raw_score(wasi_total=nonverbal, age=age)
                nonverbaliqs.append(wasi_scaled)
            
            # can't find a valid nonverbaliq
            else:
                # add to report
                _report = 'nonverbaliq not found for {}'.format(row.subject)
                
                self.report['nonverbaliq'].append(_report)
                # add to output
                nonverbaliqs.append(np.nan)
        return nonverbaliqs

    def get_bdi(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting bdi...')
        self.report['bdi']=[]
        bdi_col = 'bdi_total'

        bdis =[]
        for i, row in df_t1.iterrows():
            # bdi is collected on task day for each session, so filter for corresponding session and redcap event
            redcap_event = self.session_map_taskday[session]
            
            # get bdi as data frame
            df_bdi = df_redcap[(df_redcap['redcap_event_name']==redcap_event) & (df_redcap['narc_id']==row.subject)][bdi_col].dropna()

            # should just be one bdi value for a given session
            if len(df_bdi.unique())==1:
                bdi = df_bdi.unique()[0]
                bdis.append(bdi)
            
            # can't find a valid bdi
            else:
                # add to report
                _report = 'bdi not found for {}'.format(row.subject)
                
                self.report['bdi'].append(_report)
                # add to output
                bdis.append(np.nan)
        return bdis

    def get_smoking_status(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting smoking status...')
        self.report['smokingstatus']=[]
        smoking_col = 'ftnd_curr_cig'

        # ftnd is collected at screening and 3 month followup
        if session in ['1','2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'

        # column names in data and labels df
        suffix = '_label'
        smoking_labeled_col = smoking_col+suffix

        # get gender labels ('male', 'female', 'trans')
        df = df_redcap.copy()
        df[smoking_labeled_col] = gkutils.Redcap.data2labels(data_df=df, labels_df=df_labels, data_columns=[smoking_col], return_cols='data', suffix=suffix)

        smokingstatuses =[]
        for i, row in df_t1.iterrows():
            # get smoking status
            #--------------------------------
            df_smoking = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][smoking_labeled_col].dropna()

            # should just be one smoking status value
            if len(df_smoking.unique())==1:
                smoking = df_smoking.unique()[0]
                smokingstatuses.append(smoking)
            
            # can't find any smoking status
            elif len(df_smoking.unique())==0:
                # add to report
                _report = 'smoking status not found for {}'.format(row.subject)
                
                self.report['smokingstatus'].append(_report)
                # add to output
                smokingstatuses.append(np.nan)
            # found multiple responses
            elif len(df_smoking.unique())>1:
                # add to report
                _report = 'multiple smoking statuses found for {}'.format(row.subject)
                
                self.report['smokingstatus'].append(_report)
                # add to output
                smokingstatuses.append(np.nan)
        return smokingstatuses

    def get_ftnd(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting ftnd...')
        self.report['ftnd']=[]
        smoking_col = 'fntd_total'
        status_col = 'ftnd_curr_cig'

        # ftnd is collected at screening and 3 month followup
        if session in ['1','2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'


        # column names in data and labels df
        suffix = '_label'
        status_labeled_col = status_col+suffix

        # get smoking status labels (never, past, current)
        df = df_redcap.copy()
        df[status_labeled_col] = gkutils.Redcap.data2labels(data_df=df, labels_df=df_labels, data_columns=[status_col], return_cols='data', suffix=suffix)

        ftnds =[]
        for i, row in df_t1.iterrows():

            # get ftnd
            df_ftnd = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][smoking_col].dropna()

            # get smoking status
            df_status = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][status_labeled_col].dropna()

            # should just be one ftnd value
            if len(df_ftnd.unique())==1:
                ftnd = df_ftnd.unique()[0]
                ftnds.append(ftnd)
            
            # can't find any ftnd
            elif len(df_ftnd.unique())==0:
                # check if the subject was never a smoker
                if len(df_status.unique())==1:
                    
                    # if never a smoker, ftnd score is 0
                    if df_status.unique()[0].lower() in ['never','past']:
                        _report = 'not currently a smoker and no ftnd score found, so ftnd score=0 for {}'.format(row.subject)
                        
                        self.report['ftnd'].append(_report)
                        ftnds.append(0)
                    
                    # if they were a smoker, but no ftnd score, then add to report
                    else:
                        _report = 'ftnd score not found for {}, but said they were a smoker at one point'.format(row.subject)
                        
                        self.report['ftnd'].append(_report)
                        ftnds.append(np.nan)
                
                # no ftnd score and no clear smoking status
                else:
                    _report = 'ftnd score not found for {}'.format(row.subject)
                    
                    self.report['ftnd'].append(_report)
                    ftnds.append(np.nan)

            # found ftnd multiple responses
            elif len(df_ftnd.unique())>1:
                # add to report
                _report = 'multiple ftnd scores found for {}'.format(row.subject)
                
                self.report['smokingstatus'].append(_report)
                # add to output
                ftnds.append(np.nan)
        
        return ftnds

    def get_cig_past30(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting cig_past30...')
        self.report['cig_past30']=[]
        cig_col = 'curr_cig_past30days'
        status_col = 'curr_cig'
        redcap_event = self.session_map_scans[session]

        # column names in data and labels df
        suffix = '_label'
        status_labeled_col = status_col+suffix

        # get smoking status labels (never, past, current)
        df = df_redcap.copy()
        df[status_labeled_col] = gkutils.Redcap.data2labels(data_df=df, labels_df=df_labels, data_columns=[status_col], return_cols='data', suffix=suffix)

        cigs=[]
        for i, row in df_t1.iterrows():
            # get smoking status
            #--------------------------------
            df_cig = df[(df['redcap_event_name']==redcap_event) & (df_redcap['narc_id']==row['subject'])][cig_col].dropna()

            # get smoking status
            df_status = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][status_labeled_col].dropna()

            # should just be one smoking status value
            if len(df_cig.unique())==1:
                cig = float(df_cig.unique()[0])
                cigs.append(cig)
            
            # if not data, check if subject reports not being a smoker
            elif len(df_cig.unique())==0:
                # check if the subject was never a smoker
                if len(df_status.unique())==1:
                    if df_status.unique()[0].lower() in ['never','past']:
                        _report = 'not currently a smoker and no cig_past30 found, so cig_past30=0 for {}'.format(row.subject)
                        
                        self.report['cig_past30'].append(_report)
                        cigs.append(0)
                    else:
                        _report = 'cig_past30 not found for {}, but said they were a smoker at one point'.format(row.subject)
                        
                        self.report['cig_past30'].append(_report)
                        cigs.append(np.nan)
                else:
                    _report = 'cig_past30 not found for {}'.format(row.subject)
                    
                    self.report['cig_past30'].append(_report)
                    cigs.append(np.nan)
            elif len(df_cig.unique())>1:
                _report = 'multiple cig_past30 values found for {}'.format(row.subject)
                
                self.report['cig_past30'].append(_report)
                cigs.append(np.nan)
        return cigs

    def get_cig_num_lastuse(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting cig_num_lastuse...')
        self.report['cig_num_lastuse']=[]
        cig_col = 'curr_cig_num'
        status_col = 'curr_cig'
        date_col = 'survey_date_currdrugs_ordinal'
        redcap_event = self.session_map_scans[session]

        # column names in data and labels df
        suffix = '_label'
        status_labeled_col = status_col+suffix

        # get smoking status labels (never, past, current)
        df = df_redcap.copy()
        df[status_labeled_col] = gkutils.Redcap.data2labels(data_df=df, labels_df=df_labels, data_columns=[status_col], return_cols='data', suffix=suffix)

        cigs=[]
        for i, row in df_t1.iterrows():
            # get smoking status
            #--------------------------------
            df_cig = df[(df['redcap_event_name']==redcap_event) & (df_redcap['narc_id']==row['subject'])][cig_col].dropna()

            # get smoking status
            df_status = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][status_labeled_col].dropna()

            # should just be one smoking status value
            if len(df_cig.unique())==1:
                cig = float(df_cig.unique()[0])
                cigs.append(cig)
            
            # if not data, check if subject reports not being a smoker
            elif len(df_cig.unique())==0:
                # check if the subject was never a smoker
                if len(df_status.unique())==1:
                    if df_status.unique()[0].lower() in ['never','past']:
                        _report = 'not currently a smoker and no cig_num found, so cig_num=0 for {}'.format(row.subject)
                        
                        self.report['cig_num_lastuse'].append(_report)
                        cigs.append(0)
                    else:
                        _report = 'cig_num not found for {}, but said they were a smoker at one point'.format(row.subject)
                        
                        self.report['cig_past30'].append(_report)
                        cigs.append(np.nan)
                else:
                    _report = 'cig_num_last not found for {}'.format(row.subject)
                    
                    self.report['cig_num_lastuse'].append(_report)
                    cigs.append(np.nan)
            # if multiple values, check if one matches the scan date
            elif len(df_cig.unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[cig_col, date_col]]
                if row.date_ordinal in df_tmp[date_col].values:
                    cig = float(df_tmp[df_tmp[date_col]==row.date_ordinal][cig_col].values[0])
                    cigs.append(cig)
                # if no date matches, then leave this subject out for now
                else:
                    _report = 'multiple cig_num_last values found for {}, but none on the specified scan date'.format(row.subject)
                    
                    self.report['cig_num_lastuse'].append(_report)
                    cigs.append(np.nan)
        return cigs

    def get_thc_past30(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting thc_past30...')
        self.report['thc_past30']=[]
        thc_col = 'thc_past30days_currdrugs'
        status_col = 'used_whichdrug_currdrugs___3'
        date_col = 'survey_date_currdrugs_ordinal'
        redcap_event = self.session_map_scans[session]

        # get smoking status labels (never, past, current)
        df = df_redcap.copy()

        thcs=[]
        for i, row in df_t1.iterrows():
            # get smoking status
            #--------------------------------
            df_thc = df[(df['redcap_event_name']==redcap_event) & (df_redcap['narc_id']==row['subject'])][thc_col].dropna()

            # get smoking status
            df_status = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][status_col].dropna()

            # should just be one smoking status value
            if len(df_thc.unique())==1:
                thc = float(df_thc.unique()[0])
                thcs.append(thc)
            
            # if not data, check if subject reports not being a smoker
            elif len(df_thc.unique())==0:
                # check if the subject was never a smoker
                if len(df_status.unique())==1:
                    if df_status.unique()[0]==0:
                        _report = 'thc past30 not found, but subject reports no thc use, so thc_past30=0 for {}'.format(row.subject)
                        
                        self.report['thc_past30'].append(_report)
                        thcs.append(0)
                    else:
                        _report = 'thc_past30 not found for {}, but said they use thc'.format(row.subject)
                        
                        self.report['thc_past30'].append(_report)
                        thcs.append(np.nan)
                else:
                    _report = 'thc_past30 not found for {}'.format(row.subject)
                    
                    self.report['thc_past30'].append(_report)
                    thcs.append(np.nan)
            # if multiple values, check if one matches the scan date
            elif len(df_thc.unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[thc_col, date_col]]
                if row.date_ordinal in df_tmp[date_col].values:
                    thc = float(df_tmp[df_tmp[date_col]==row.date_ordinal][thc_col].values[0])
                    thcs.append(thc)

                # if no date matches, then leave this subject out for now
                else:
                    _report = 'multiple thc_past30 values found for {}, but none on the specified scan date'.format(row.subject)
                    
                    self.report['thc_past30'].append(_report)
                    thcs.append(np.nan)

        return thcs
    
    def get_thc_years_reguse(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting thc_years_reguse...')
        self.report['thc_years_reguse']=[]
        thc_col = 'asi_thc_reg_dur_yrs'
        ao_col = 'asi_thc_ao_reg'
        prog_col = 'asi_thc_progression'
        status_col = 'used_whichdrug_currdrugs___3'
        if session in ['1','2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'

        direct_map={}

        thcs=[]
        for i, row in df_t1.iterrows():
            # get smoking status
            #--------------------------------
            df_thc = df_redcap[(df_redcap['redcap_event_name']==redcap_event) & (df_redcap['narc_id']==row['subject'])][thc_col].dropna().astype(str)

            df_ao = df_redcap[(df_redcap['redcap_event_name']==redcap_event) & (df_redcap['narc_id']==row['subject'])][ao_col].dropna()

            # get smoking status
            df_status = df_redcap[(df_redcap['redcap_event_name']==redcap_event) & (df_redcap['narc_id']==row['subject'])][status_col].dropna()

            # get progression of thc use
            df_prog = df_redcap[(df_redcap['redcap_event_name']==redcap_event) & (df_redcap['narc_id']==row['subject'])][prog_col].dropna()
            # did we record any details on use progression?
            if len(df_prog.unique())==1:
                prog = df_prog.unique()[0]
            else:
                prog = np.nan

            # if one value is found
            if len(df_thc.unique())==1:

                # get normalized duration values from string responses (need to tighten this up when data is entered)
                thc_norm, unhandled = self.normalize_duration(input_series=df_thc, new_series=df_thc.copy(), direct_map=direct_map,new_val_max=None, output_type='years', default_units='years')

                # if a numeric value is returned with no undandled cases add to list
                if len(unhandled)==0 and isinstance(thc_norm.values[0], (float,int)):
                    thcs.append(thc_norm.values[0])
                
                # otherwise report subject response and age of onset
                elif len(unhandled)>0:
                    _report = 'subject: {}, thc years reguse not found. they gave the follwoing responses for years of regular use: {}, and age of onset:{}, and progression: {}'.format(row.subject, df_thc.unique()[0], df_ao.unique()[0], prog)
                    
                    self.report['thc_years_reguse'].append(_report)
                    thcs.append(np.nan)

            
            # if not data, check if subject reports not being a smoker
            elif len(df_thc.unique())==0:
                # check if the subject endorsed cannabis
                if len(df_status.unique())==1:
                    
                    # if not, set thc_years_reguse to 0
                    if df_status.unique()[0]==0:
                        _report = 'subject: {}, thc years reguse not found, but subject reports no thc use, so thc_years_reguse=0'.format(row.subject)
                        
                        self.report['thc_years_reguse'].append(_report)
                        thcs.append(0)
                    
                    # otherwise check if they reported an age of onset
                    elif len(df_ao.unique())==1:
                        ao = float(df_ao.unique()[0])
                        thc = row.age - ao
                        thcs.append(thc)
                        _report = 'subject: {}, thc_years_reguse not found, but they reported an age of onset of {}, so thc_years_reguse={} based on reported age of onset ({}) and age at scan ({}). check notes on progression match this: {} '.format(row.subject, ao, thc, ao, row.age, prog)
                        
                        self.report['thc_years_reguse'].append(_report)
                    
                    # if they reported using thc but not an age of onset, assume they never became regular
                    else:
                        _report = 'subject: {}, thc_years_reguse not found. no age of onset for regular use could be found either. but said they use thc. assume thc use never became regular and set to 0. double check that this matches progression notes: {}'.format(row.subject, prog)
                        
                        self.report['thc_years_reguse'].append(_report)
                        thcs.append(0)
                
                # no clear status information
                else:
                    _report = 'thc_years_reguse not found for {}'.format(row.subject)
                    
                    self.report['thc_years_reguse'].append(_report)
                    thcs.append(np.nan)
            
            # if multiple values are found
            elif len(df_thc.unique())>1:
                _report = 'multiple thc_years_reguse values found for {}'.format(row.subject)
                
                self.report['thc_years_reguse'].append(_report)
                thcs.append(np.nan)
        return thcs

    def get_alc_past30(self, df_t1, df_redcap, df_labels, session):
        ''' past 30 days alcohol use is in the asi 
        '''
        print('getting alc_past30...')
        self.report['alc_past30']=[]
        alc_col = 'asi_alc_pastmonth'
        status_col = 'asi_alc_hx' # 1=yes drinker, 2=not drinker
        if session in ['1','2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'

        df = df_redcap.copy()
        days=[]
        for i, row in df_t1.iterrows():
            
            df_drink = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][alc_col].dropna()

            df_status = df[ (df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][status_col].dropna()

            # if one value is found
            if len(df_drink.unique())==1:
                days.append(df_drink.unique()[0])
            
            # if no data is found check if subject reports not being a drinker
            elif len(df_drink.unique())==0:
                # is there a clear drinking status/endorsement?
                if len(df_status.unique())==1:

                    # if not a drinker, set alc_past30 to 0
                    if df_status.unique()[0]==2:
                        _report = 'subject: {}, alc_past30 not found, but subject reports no alcohol use, so alc_past30=0'.format(row.subject)
                        
                        self.report['alc_past30'].append(_report)
                        days.append(0)
                    # if reported drinking, but no response set alc_past30 to np.nan
                    elif df_status.unique()[0]==1:
                        _report = 'subject: {}, alc_past30 not found, but subject reports alcohol use, so alc_past30=np.nan'.format(row.subject)
                        
                        self.report['alc_past30'].append(_report)
                        days.append(np.nan)
                else:
                    _report = 'subject: {}, alc_past30 not found, and no clear drinking status could be found'.format(row.subject)
                    
                    self.report['alc_past30'].append(_report)
                    days.append(np.nan)
            
            # multiple values found
            elif len(df_drink.unique())>1:
                _report = 'subject: {}, multiple alc_past30 values found'.format(row.subject)
                
                self.report['alc_past30'].append(_report)
                days.append(np.nan)
        return days

    def get_alc_years_reguse(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting alc_years_reguse...')
        self.report['alc_years_reguse']=[]
        alc_col = 'asi_alc_reg_dur_yrs'
        ao_col = 'asi_thc_ao_reg'
        status_col = 'asi_alc_hx'
        prog_col = 'asi_alc_progression'

        # we only have asi data for screening and 3 month followup
        if session in ['1','2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'

        direct_map={}
        df = df_redcap.copy()
        years=[]
        for i, row in df_t1.iterrows():
            # get smoking status
            #--------------------------------
            df_alc = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][alc_col].dropna().astype(str)

            df_ao = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][ao_col].dropna()

            # get smoking status
            df_status = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][status_col].dropna()

            # get progression notes
            df_prog = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][prog_col].dropna().astype(str)
            # did we record any details on use progression?
            if len(df_prog.unique())==1:
                prog = df_prog.unique()[0]
            else:
                prog = np.nan

            # if one value is found
            if len(df_alc.unique())==1:

                # get normalized duration values from string responses (need to tighten this up when data is entered)
                alc_norm, unhandled = self.normalize_duration(input_series=df_alc, new_series=df_alc.copy(), direct_map=direct_map,new_val_max=None, output_type='years', default_units='years')

                # if a numeric value is returned with no undandled cases add to list
                if len(unhandled)==0 and isinstance(alc_norm.values[0], (float,int)):
                    years.append(alc_norm.values[0])
                
                # otherwise report subject response and age of onset
                elif len(unhandled)>0:
                    if len(df_ao.unique())==1:
                        ao= df_ao.unique()[0]
                    else:
                        ao = np.nan
                    _report = 'subject: {}, alc years reguse not found. they gave the follwoing responses for years of regular use: {}, and age of onset: {}, and progression: {}'.format(row.subject, df_alc.unique()[0], ao, prog)
                    
                    self.report['alc_years_reguse'].append(_report)
                    years.append(np.nan)

            
            # if not data, check if subject reports not being a smoker
            elif len(df_alc.unique())==0:
                # check if the subject endorsed alcohol use
                if len(df_status.unique())==1:
                    
                    # if not, set alc_years_reguse to 0
                    if df_status.unique()[0]==0:
                        _report = 'subject: {}, alc years reguse not found, but subject reports no alc use, so alc_years_reguse=0. double check progression notes: {}'.format(row.subject, prog)
                        
                        self.report['alc_years_reguse'].append(_report)
                        years.append(0)
                    
                    # otherwise check if they reported an age of onset
                    elif len(df_ao.unique())==1:
                        ao = float(df_ao.unique()[0])
                        _years = row.age - ao
                        years.append(_years)
                        _report = 'subject: {}, alc_years_reguse not found, but they reported an age of onset of {}, so alc_years_reguse={} based on reported age of onset ({}) and age at scan ({}). check that this matches progression notes: {}'.format(row.subject, ao, _years, ao, row.age, prog)
                        
                        self.report['alc_years_reguse'].append(_report)
                    
                    # if they reported using alc but not an age of onset, assume they never became regular
                    else:
                        _report = 'subject: {}, alc_years_reguse not found. no age of onset for regular use could be found either. but said they use alc. assume alc use never became regular and set to 0. check that this matches progression notes: {}'.format(row.subject, prog)
                        
                        self.report['alc_years_reguse'].append(_report)
                        years.append(0)
                
                # no clear status information
                else:
                    _report = 'subject: {}, alc_years_reguse not found'.format(row.subject)
                    
                    self.report['alc_years_reguse'].append(_report)
                    years.append(np.nan)
            
            # if multiple values are found
            elif len(df_alc.unique())>1:
                _report = 'subject: {}, multiple alc_years_reguse values found'.format(row.subject)
                
                self.report['alc_years_reguse'].append(_report)
                years.append(np.nan)
        return years

    def get_smast(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting smast...')
        self.report['smast']=[]
        col = 'smast_total'
        status_col = 'asi_alc_hx' # 1=yes drinker, 2=not drinker
        if session in ['1','2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'
        
        df = df_redcap.copy()
        smasts=[]
        for i, row in df_t1.iterrows():
            df_smast = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][col].dropna()

            df_status  = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][status_col].dropna()

            # if one value is found
            if len(df_smast.unique())==1:
                smasts.append(df_smast.values[0])

            # no score found, check if subject reports not being a drinker
            elif len(df_smast.unique())==0:
                # check for drinking status
                if len(df_status.unique())==1:

                    # not a drinker, set smast to 0
                    if df_status.unique()[0]==2:
                        _report = 'subject: {}, smast not found, but subject reports no alc use, so smast=0'.format(row.subject)
                        
                        self.report['smast'].append(_report)
                        smasts.append(0)
                    # is a drinker, but no smast, return nan
                    else:
                        _report = 'subject: {}, smast not found, but subject does report drinking'.format(row.subject)
                        
                        self.report['smast'].append(_report)
                        smasts.append(np.nan)

            elif len(df_smast.unique())>1:
                _report = 'subject: {}, multiple smast values found'.format(row.subject)
                
                self.report['smast'].append(_report)
                smasts.append(np.nan)
        return smasts
    
    def get_hcq(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting hcq...')
        self.report['hcq']=[]
        col = 'hcq_total'
        status_col = 'asi_opioid_hx' # 1=yes user, 2=not user

        # ASI and HCQ are only collected at screening and 3 month followup
        if session in ['1','2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'

        df = df_redcap.copy()
        hcqs=[]
        for i, row in df_t1.iterrows():

            # skip control for drug variables
            if row.group.lower() in ['hc','control','ctl']:
                hcqs.append(np.nan)
                continue

            df_hcq = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][col].dropna()

            df_status  = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][status_col].dropna()
            
            # if one value is found
            if len(df_hcq.unique())==1:
                hcqs.append(df_hcq.values[0])

            # no score found, check if subject reports not being a user
            elif len(df_hcq.unique())==0:
                # check for status
                if len(df_status.unique())==1:

                    # not a user, set hcq to 0
                    if df_status.unique()[0]==2:
                        _report = 'subject: {}, hcq not found, but subject reports no opioid use, so hcq=0'.format(row.subject)
                        
                        self.report['hcq'].append(_report)
                        hcqs.append(0)
                    # is a user, but no hcq, return nan
                    else:
                        _report = 'subject: {}, hcq not found, but subject does report opioid use'.format(row.subject)
                        
                        self.report['hcq'].append(_report)
                        hcqs.append(np.nan)

            elif len(df_hcq.unique())>1:
                _report = 'subject: {}, multiple hcq values found'.format(row.subject)
                
                self.report['hcq'].append(_report)
                hcqs.append(np.nan)

        return hcqs

    def get_heroin_past30(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting heroin_past30...')
        self.report['heroin_past30']=[]
        her_col = 'heroin_past30days_currdrugs'
        status_col = 'heroin_currdrugs' # 0=no, 1=yes
        date_col = 'survey_date_currdrugs_ordinal'
        redcap_event = self.session_map_scans[session]

        df = df_redcap.copy()
        heroin_past30s=[]
        for i, row in df_t1.iterrows():
            # skip control for drug variables
            if row.group.lower() in ['hc','control','ctl']:
                heroin_past30s.append(np.nan)
                continue

            df_heroin_past30 = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][her_col].dropna()

            df_status  = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][status_col].dropna()
            
            # if one value is found
            if len(df_heroin_past30.unique())==1:
                heroin_past30s.append(df_heroin_past30.unique()[0])

            # no score found, check if subject reports not being a user
            elif len(df_heroin_past30.unique())==0:
                # check for status
                if len(df_status.unique())==1:

                    # not a user, set heroin_past30 to 0
                    if df_status.unique()[0]==0 or row.group.lower() in ['hc', 'control', 'ctl']:
                        _report = 'subject: {}, heroin_past30 not found, but subject reports no heroin use, so heroin_past30=0'.format(row.subject)
                        
                        self.report['heroin_past30'].append(_report)
                        heroin_past30s.append(0)
                    # is a user, but no heroin_past30, return nan
                    else:
                        _report = 'subject: {}, heroin_past30 not found, but subject does report heroin use'.format(row.subject)
                        
                        self.report['heroin_past30'].append(_report)
                        heroin_past30s.append(np.nan)

            elif len(df_heroin_past30.unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[her_col, date_col]]
                if row.date_ordinal in df_tmp[date_col].values:
                    heroin_past30 = float(df_tmp[df_tmp[date_col]==row.date_ordinal][her_col].values[0])
                    heroin_past30s.append(heroin_past30)
                    _report = 'subject: {}, multiple heroin_past30 values found, but only one matches scan date, so heroin_past30={}'.format(row.subject, heroin_past30)
                else:
                    _report = 'subject: {}, multiple heroin_past30 values found'.format(row.subject)
                    
                    self.report['heroin_past30'].append(_report)
                    heroin_past30s.append(np.nan)

        return heroin_past30s

    def get_heroin_years_reguse(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting heroin_years_reguse...')
        self.report['heroin_years_reguse']=[]
        her_col = 'asi_opioid_reg_dur_yrs'
        ao_col = 'asi_opioid_ao_reg'
        status_col = 'asi_opioid_hx' # 1=yes, 2=no
        prog_col = 'asi_opioid_progression'
        date_col = 'survey_date_asi_ordinal'
        assume_continuous_use = True # if true, calculate missing values based on age of onset (ie assume they have been continuously since onset )
        if session in ['1', '2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'

        direct_map={}
        df = df_redcap.copy()
        heroin_years_reguses=[]
        for i, row in df_t1.iterrows():
            # skip control for drug variables
            if row.group.lower() in ['hc','control','ctl']:
                heroin_years_reguses.append(np.nan)
                continue


            df_her = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][her_col].dropna().astype(str)

            df_status  = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][status_col].dropna()

            df_ao = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][ao_col].dropna()

            df_prog = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][prog_col].dropna()
            # did we record any details on use progression?
            if len(df_prog.unique())==1:
                prog = df_prog.unique()[0]
            else:
                prog = np.nan
            
            # if one value is found
            if len(df_her.unique())==1:
                # get normalized duration values from string responses (need to tighten this up when data is entered)
                her_norm, unhandled = self.normalize_duration(input_series=df_her, new_series=df_her.copy(), direct_map=direct_map,new_val_max=None, output_type='years', default_units='years')

                # if a numeric value is returned with no undandled cases add to list
                if len(unhandled)==0 and isinstance(her_norm.values[0], (float,int)):
                    heroin_years_reguses.append(her_norm.values[0])
                
                # otherwise report subject response and age of onset
                elif len(unhandled)>0:
                    # do they report an age of onset?
                    if len(df_ao.unique())==1:
                        ao= df_ao.unique()[0]
                    else:
                        ao = np.nan

                    _report = 'subject: {},heroin years reguse not found. they gave the follwoing responses for years of regular use: {},age of onset: {}, and progression of use: {}'.format(row.subject, df_her.unique()[0], ao, prog)
                    self.report['heroin_years_reguse'].append(_report)
                    heroin_years_reguses.append(np.nan)

            # no score found, check if subject reports not being a user
            elif len(df_her.unique())==0:
                # check for status
                if len(df_status.unique())==1:

                    # not a user, set heroin_years_reguse to 0
                    if df_status.unique()[0]==2:
                        _report = 'subject: {}, heroin_years_reguse not found, but subject reports no heroin use, so heroin_years_reguse=0'.format(row.subject)
                        self.report['heroin_years_reguse'].append(_report)
                        heroin_years_reguses.append(0)

                    # otherwise check if they reported an age of onset and infer from age of onset
                    elif len(df_ao.unique())==1:
                        ao = float(df_ao.unique()[0])
                        _years = row.age - ao

                        if assume_continuous_use and isinstance(prog, float) and np.isnan(prog):
                            heroin_years_reguses.append(_years)
                            _report = 'subject: {}, heroin_years_reguse not found, but they reported an age of onset of {}, so heroin_years_reguse={} based on reported age of onset ({}) and age at scan ({}). double check notes on progression of use:{}'.format(row.subject, ao, _years, ao, row.age, prog)
                        else:
                            heroin_years_reguses.append(np.nan)
                            _report = 'subject: {}, heroin_years_reguse not found, but they reported an age of onset of {}, but we cannot assume abstinence from then till time of scan. double check notes on progression of use:{}'.format(row.subject, ao, prog)
                        self.report['heroin_years_reguse'].append(_report)

                    # is a user, but no heroin_years_reguse, return nan
                    else:
                        _report = 'subject: {}, heroin_years_reguse not found, but subject does report heroin use. see notes on progression of use: {}'.format(row.subject, prog)
                        self.report['heroin_years_reguse'].append(_report)
                        heroin_years_reguses.append(np.nan)

            elif len(df_her.unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[her_col, date_col]]
                if row.date_ordinal in df_tmp[date_col].values:
                    heroin_years_reguse = float(df_tmp[df_tmp[date_col]==row.date_ordinal][her_col].values[0])
                    heroin_years_reguses.append(heroin_years_reguse)
                    _report = 'subject: {}, multiple heroin_years_reguse values found, but only one matches scan date, so heroin_years_reguse={}'.format(row.subject, heroin_years_reguse)
                    self.report['heroin_years_reguse'].append(_report)
                else:
                    _report = 'subject: {}, multiple heroin_years_reguse values found, but none match scan date, so heroin_years_reguse=np.nan'.format(row.subject)
                    heroin_years_reguses.append(np.nan)
                    self.report['heroin_years_reguse'].append(_report)
        return heroin_years_reguses

    def get_sds(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting sds')
        self.report['sds'] = []
        sds_col = 'sds_total'
        status_col = 'asi_opioid_hx' # 1=yes, 2=no
        date_col = 'sds_date_ordinal'
        if session in ['1','2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'

        df = df_redcap.copy()
        sds = []
        for i, row in df_t1.iterrows():
            # get sds
            df_sds = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[sds_col]].dropna()

            # get status as user/non-user
            df_status = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[status_col]].dropna()
            if len(df_status[status_col].unique())==1:
                status = df_status[status_col].unique().astype(float)[0]
            else:
                status = np.nan

            # if there is a value for sds, add it to the list
            if len(df_sds[sds_col].unique())==1:
                sds.append(df_sds[sds_col].unique()[0])
            
            # if no value found check if they reported not being a user
            elif len(df_sds[sds_col].unique())==0:
                # not a user
                if status==2:
                    sds.append(0)
                    _report = 'subject: {}, sds not found, but subject reports no opioid use, so sds=0'.format(row.subject)
                    self.report['sds'].append(_report)
                # no value, but is a user
                else:
                    sds.append(np.nan)
                    _report = 'subject: {}, sds not found, but subject reports opioid use, so sds=np.nan'.format(row.subject)
                    self.report['sds'].append(_report)

            # multiple values found
            elif len(df_sds[sds_col].unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[sds_col, date_col]]
                # matching date found, add to list
                if row.date_ordinal in df_tmp[date_col].values:
                    sds.append(float(df_tmp[df_tmp[date_col]==row.date_ordinal][sds_col].values[0]))
                    _report = 'subject: {}, multiple sds values found, but only one matches scan date, so sds={}'.format(row.subject, sds[-1])
                    self.report['sds'].append(_report)
                # no matching date found, add nan
                else:
                    sds.append(np.nan)
                    _report = 'subject: {}, multiple sds values found, but none match scan date, so sds=np.nan'.format(row.subject)
                    self.report['sds'].append(_report)

        return sds

    def get_sows(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting sows')
        self.report['sows'] = []
        sows_col = 'sows_total'
        status_col = 'asi_opioid_hx' # 1=yes, 2=no
        date_col = 'sows_date_ordinal'   
        if session in ['1','2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'
        
        df = df_redcap.copy()
        sows = []
        for i, row in df_t1.iterrows():
            # get sows
            df_sows = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[sows_col]].dropna()

            # get status as user/non-user
            df_status = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[status_col]].dropna()
            if len(df_status[status_col].unique())==1:
                status = df_status[status_col].unique().astype(float)[0]
            else:
                status = np.nan

            # if there is a value for sows, add it to the list
            if len(df_sows[sows_col].unique())==1:
                sows.append(df_sows[sows_col].unique()[0])
            
            # if no value found check if they reported not being a user
            elif len(df_sows[sows_col].unique())==0:
                # not a user
                if status==2:
                    sows.append(0)
                    _report = 'subject: {}, sows not found, but subject reports no opioid use, so sows=0'.format(row.subject)
                    self.report['sows'].append(_report)
                # no value, but is a user
                else:
                    sows.append(np.nan)
                    _report = 'subject: {}, sows not found, but subject reports opioid use, so sows=np.nan'.format(row.subject)
                    self.report['sows'].append(_report)

            # multiple values found
            elif len(df_sows[sows_col].unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[sows_col, date_col]]
                # matching date found, add to list
                if row.date_ordinal in df_tmp[date_col].values:
                    sows.append(float(df_tmp[df_tmp[date_col]==row.date_ordinal][sows_col].values[0]))
                    _report = 'subject: {}, multiple sows values found, but only one matches scan date, so sows={}'.format(row.subject, sows[-1])
                    self.report['sows'].append(_report)
                # no matching date found, add nan
                else:
                    sows.append(np.nan)
                    _report = 'subject: {}, multiple sows values found, but none match scan date, so sows=np.nan'.format(row.subject)
                    self.report['sows'].append(_report)
        return sows

    def get_heroin_abstinence(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting heroin_abstinence')
        self.report['heroin_abstinence'] = []
        her_col = 'curr_her_abs'
        status_col = 'asi_opioid_hx' # 1=yes, 2=no
        date_col = 'survey_date_currdrugs_ordinal'
        redcap_event = self.session_map_scans[session]
        if session in ['1','2']:
            status_redcap_event = 'screening_arm_1'
        elif session in ['3']:
            status_redcap_event = '3_month_followup_arm_1'

        df = df_redcap.copy()
        heroin_abstinence = []
        for i, row in df_t1.iterrows():
            # skip control for drug variables
            if row.group.lower() in ['hc','control','ctl']:
                heroin_abstinence.append(np.nan)
                continue
            # get heroin_abstinence
            df_her = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[her_col]].dropna()

            # get status as user/non-user
            df_status = df[(df['redcap_event_name']==status_redcap_event) & (df['narc_id']==row['subject'])][[status_col]].dropna()
            if len(df_status[status_col].unique())==1:
                status = df_status[status_col].unique().astype(float)[0]
            else:
                status = np.nan

            # if there is a value for heroin_abstinence, add it to the list
            if len(df_her[her_col].unique())==1:
                heroin_abstinence.append(float(df_her[her_col].unique()[0]))
            
            # if no value found check if they reported not being a user
            elif len(df_her[her_col].unique())==0:
                # not a user
                if status==2:
                    heroin_abstinence.append(0)
                    _report = 'subject: {}, heroin_abstinence not found, but subject reports no opioid use, so heroin_abstinence=0'.format(row.subject)
                    self.report['heroin_abstinence'].append(_report)
                # no value, but is a user
                else:
                    heroin_abstinence.append(np.nan)
                    _report = 'subject: {}, heroin_abstinence not found, but subject reports opioid use, so heroin_abstinence=np.nan'.format(row.subject)
                    self.report['heroin_abstinence'].append(_report)

            # multiple values found
            elif len(df_her[her_col].unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[her_col, date_col]]
                # matching date found, add to list
                if row.date_ordinal in df_tmp[date_col].values:
                    heroin_abstinence.append(float(df_tmp[df_tmp[date_col]==row.date_ordinal][her_col].values[0]))
                    _report = 'subject: {}, multiple heroin_abstinence values found, but only one matches scan date, so heroin_abstinence={}'.format(row.subject, heroin_abstinence[-1])
                    self.report['heroin_abstinence'].append(_report)
                # no matching date found, add nan
                else:
                    heroin_abstinence.append(np.nan)
                    _report = 'subject: {}, multiple heroin_abstinence values found, but none match scan date, so heroin_abstinence=np.nan'.format(row.subject)
                    self.report['heroin_abstinence'].append(_report)
        return heroin_abstinence
                
    def get_methadone_dose(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting methadone_dose')
        self.report['methadone_dose'] = []
        meth_col = 'curr_meds_vit_list'
        status_col = 'asi_opioid_hx' # 1=yes, 2=no
        date_col = 'survey_date_currdrugs_ordinal'
        redcap_event = self.session_map_scans[session]
        if session in ['1','2']:
            status_redcap_event = 'screening_arm_1'
        elif session in ['3']:
            status_redcap_event = '3_month_followup_arm_1'
        
        df = df_redcap.copy()
        methadone_dose = []
        for i, row in df_t1.iterrows():
            # skip control for drug variables
            if row.group.lower() in ['hc','control','ctl']:
                methadone_dose.append(np.nan)
                continue

            # get methadone_dose
            df_meth = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[meth_col]].dropna()

            # get status as user/non-user
            df_status = df[(df['redcap_event_name']==status_redcap_event) & (df['narc_id']==row['subject'])][[status_col]].dropna()
            if len(df_status[status_col].unique())==1:
                status = df_status[status_col].unique().astype(float)[0]
            else:
                status = np.nan
            
            # if there is a value for methadone_dose, add it to the list
            if len(df_meth[meth_col].unique())==1:
                medication_string = df_meth[meth_col].unique()[0]
                dose = self.get_medication_dose(string=medication_string, medication='methadone', )
                methadone_dose.append(dose)

            
            # if no value found check if they reported not being a user
            elif len(df_meth[meth_col].unique())==0:
                # not a user
                if status==2:
                    methadone_dose.append(np.nan)
                    _report = 'subject: {}, methadone_dose not found, but subject reports no opioid use, so methadone_dose=0'.format(row.subject)
                    self.report['methadone_dose'].append(_report)
                # no value, but is a user
                else:
                    methadone_dose.append(np.nan)
                    _report = 'subject: {}, methadone_dose not found, but subject reports opioid use, so methadone_dose=np.nan'.format(row.subject)
                    self.report['methadone_dose'].append(_report)

            # multiple values found
            elif len(df_meth[meth_col].unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[meth_col, date_col]]
                # matching date found, add to list
                if row.date_ordinal in df_tmp[date_col].values:
                    medication_string = df_tmp[df_tmp[date_col]==row.date_ordinal][meth_col].unique()[0]
                    dose = self.get_medication_dose(string=medication_string, medication='methadone', )
                    methadone_dose.append(dose)
                    _report = 'subject: {}, multiple methadone_dose values found, but only one matches scan date, so methadone_dose={}'.format(row.subject, methadone_dose[-1])
                    self.report['methadone_dose'].append(_report)
                # no matching date found, add nan
                else:
                    methadone_dose.append(np.nan)
                    _report = 'subject: {}, multiple methadone_dose values found, but none match scan date, so methadone_dose=np.nan'.format(row.subject)
                    self.report['methadone_dose'].append(_report)

        return methadone_dose

    def get_suboxone_dose(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting suboxone_dose')
        self.report['suboxone_dose'] = []
        meth_col = 'curr_meds_vit_list'
        status_col = 'asi_opioid_hx' # 1=yes, 2=no
        date_col = 'survey_date_currdrugs_ordinal'
        redcap_event = self.session_map_scans[session]
        if session in ['1','2']:
            status_redcap_event = 'screening_arm_1'
        elif session in ['3']:
            status_redcap_event = '3_month_followup_arm_1'
        
        df = df_redcap.copy()
        suboxone_dose = []
        for i, row in df_t1.iterrows():
            # skip control for drug variables
            if row.group.lower() in ['hc','control','ctl']:
                suboxone_dose.append(np.nan)
                continue
            # get suboxone_dose
            df_meth = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[meth_col]].dropna()

            # get status as user/non-user
            df_status = df[(df['redcap_event_name']==status_redcap_event) & (df['narc_id']==row['subject'])][[status_col]].dropna()
            if len(df_status[status_col].unique())==1:
                status = df_status[status_col].unique().astype(float)[0]
            else:
                status = np.nan
            
            # if there is a value for suboxone_dose, add it to the list
            if len(df_meth[meth_col].unique())==1:
                medication_string = df_meth[meth_col].unique()[0]
                dose = self.get_medication_dose(string=medication_string, medication='suboxone', )
                suboxone_dose.append(dose)

            
            # if no value found check if they reported not being a user
            elif len(df_meth[meth_col].unique())==0:
                # not a user
                if status==2:
                    suboxone_dose.append(np.nan)
                    _report = 'subject: {}, suboxone_dose not found, but subject reports no opioid use, so suboxone_dose=0'.format(row.subject)
                    self.report['suboxone_dose'].append(_report)
                # no value, but is a user
                else:
                    suboxone_dose.append(np.nan)
                    _report = 'subject: {}, suboxone_dose not found, but subject reports opioid use, so suboxone_dose=np.nan'.format(row.subject)
                    self.report['suboxone_dose'].append(_report)

            # multiple values found
            elif len(df_meth[meth_col].unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[meth_col, date_col]]
                # matching date found, add to list
                if row.date_ordinal in df_tmp[date_col].values:
                    medication_string = df_tmp[df_tmp[date_col]==row.date_ordinal][meth_col].unique()[0]
                    dose = self.get_medication_dose(string=medication_string, medication='suboxone', )
                    suboxone_dose.append(dose)
                    _report = 'subject: {}, multiple suboxone_dose values found, but only one matches scan date, so suboxone_dose={}'.format(row.subject, suboxone_dose[-1])
                    self.report['suboxone_dose'].append(_report)
                # no matching date found, add nan
                else:
                    suboxone_dose.append(np.nan)
                    _report = 'subject: {}, multiple suboxone_dose values found, but none match scan date, so suboxone_dose=np.nan'.format(row.subject)
                    self.report['suboxone_dose'].append(_report)
                    
        return suboxone_dose
    
    def get_medication_type(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting medication_type')
        self.report['medication_type'] = []
        meth_col = 'curr_meds_vit_list'
        status_col = 'asi_opioid_hx' # 1=yes, 2=no
        date_col = 'survey_date_currdrugs_ordinal'
        redcap_event = self.session_map_scans[session]
        if session in ['1','2']:
            status_redcap_event = 'screening_arm_1'
        elif session in ['3']:
            status_redcap_event = '3_month_followup_arm_1'
        
        df = df_redcap.copy()
        medications = []
        for i, row in df_t1.iterrows():
            # skip control for drug variables
            if row.group.lower() in ['hc','control','ctl']:
                medications.append(np.nan)
                continue
                
            # get medications
            df_meth = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[meth_col]].dropna()

            # get status as user/non-user
            df_status = df[(df['redcap_event_name']==status_redcap_event) & (df['narc_id']==row['subject'])][[status_col]].dropna()
            if len(df_status[status_col].unique())==1:
                status = df_status[status_col].unique().astype(float)[0]
            else:
                status = np.nan
            
            # if there is a value for medications, add it to the list
            if len(df_meth[meth_col].unique())==1:
                medication_string = df_meth[meth_col].unique()[0]
                medtype = self.get_medications(string=medication_string)
                medications.append(medtype)

            
            # if no value found check if they reported not being a user
            elif len(df_meth[meth_col].unique())==0:
                # not a user
                if status==2:
                    medications.append(np.nan)
                    _report = 'subject: {}, medications not found, but subject reports no opioid use, so medications=0'.format(row.subject)
                    self.report['medication_type'].append(_report)
                # no value, but is a user
                else:
                    medications.append(np.nan)
                    _report = 'subject: {}, medications not found, but subject reports opioid use, so medications=np.nan'.format(row.subject)
                    self.report['medication_type'].append(_report)

            # multiple values found
            elif len(df_meth[meth_col].unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[meth_col, date_col]]
                # matching date found, add to list
                if row.date_ordinal in df_tmp[date_col].values:
                    medication_string = df_tmp[df_tmp[date_col]==row.date_ordinal][meth_col].unique()[0]
                    medtype = self.get_medications(string=medication_string )
                    medications.append(medtype)
                    _report = 'subject: {}, multiple medications values found, but only one matches scan date, so medications={}'.format(row.subject, medications[-1])
                    self.report['medication_type'].append(_report)
                # no matching date found, add nan
                else:
                    medications.append(np.nan)
                    _report = 'subject: {}, multiple medications values found, but none match scan date, so medications=np.nan'.format(row.subject)
                    self.report['medication_type'].append(_report)

        return medications

    def get_heroin_roa(self, df_t1, df_redcap, df_labels, session):
        '''
        '''
        print('getting heroin_roa')
        self.report['heroin_roa'] = []
        roa_col = 'asi_opioid_heroin_roa'
        date_col = 'survey_date_asi_ordinal'
        if session in ['1','2']:
            redcap_event = 'screening_arm_1'
        elif session in ['3']:
            redcap_event = '3_month_followup_arm_1'

        df = df_redcap.copy()
        # get roa labels
        # --------------------------------
        suffix = '_label'
        label_col = roa_col+suffix
        df = df_redcap.copy()
        df[label_col] = gkutils.Redcap.data2labels(data_df=df, labels_df=df_labels, data_columns=[roa_col], return_cols='data', suffix=suffix)

        roas = []
        for i, row in df_t1.iterrows():
            # skip control for drug variables
            if row.group.lower() in ['hc','control','ctl']:
                roas.append(np.nan)
                continue

            # if not heroin user, add nan
            if row.group.lower() in ['hc','control','controls', 'ctl']:
                roas.append(np.nan)
                continue
            
            # get roa
            df_roa = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[roa_col, label_col]].dropna()
            
            # if there is a value for roa, add it to the list
            if len(df_roa[roa_col].unique())==1:
                roa = df_roa[label_col].unique()[0]
                roas.append(roa)
            
            # if no value found, add nan
            elif len(df_roa[roa_col].unique())==0:
                roas.append(np.nan)
                _report = 'subject: {}, heroin_roa not found, so heroin_roa=np.nan'.format(row.subject)
                self.report['heroin_roa'].append(_report)
            # multiple values found
            elif len(df_roa[roa_col].unique())>1:
                # check the dates of surveys and see if one matches the scan date
                df_tmp = df[(df['redcap_event_name']==redcap_event) & (df['narc_id']==row['subject'])][[roa_col, label_col, date_col]]
                # matching date found, add to list
                if row.date_ordinal in df_tmp[date_col].values:
                    roa = df_tmp[df_tmp[date_col]==row.date_ordinal][label_col].unique()[0]
                    roas.append(roa)
                    _report = 'subject: {}, multiple heroin_roa values found, but only one matches scan date, so heroin_roa={}'.format(row.subject, roas[-1])
                    self.report['heroin_roa'].append(_report)
                # no matching date found, add nan
                else:
                    roas.append(np.nan)
                    _report = 'subject: {}, multiple heroin_roa values found, but none match scan date, so heroin_roa=np.nan'.format(row.subject)
                    self.report['heroin_roa'].append(_report)

        return roas

    def get_medications(cls, string, join_string='; '):
        '''
        '''
        possible_medications = [
            'methadone',
            'buprenorphine',
            'naltrexone',
            'naloxone',
            'suboxone',
        ]
        medications=[]
        for med in possible_medications:
            if med in string.lower():
                medications.append(med)
        if len(medications)==0:
            output_string = np.nan
        else:
            output_string = join_string.join(medications)
        return output_string

    def get_medication_dose(cls, string, medication='methadone', ):
        '''
        '''
        numeric_pattern = r"\d+(?:\.\d+)?"
        if medication=='methadone':
            pattern = r"methadone\s*([^a-zA-Z]*\d+(?:\.\d+)?)"
        elif medication=='buprenorphine':
            pattern = r"buprenorphine\s*([^a-zA-Z]*\d+(?:\.\d+)?)"
        elif medication=='naltrexone':
            pattern = r"naltrexone\s*([^a-zA-Z]*\d+(?:\.\d+)?)"
        elif medication=='naloxone':
            pattern = r"naloxone\s*([^a-zA-Z]*\d+(?:\.\d+)?)"
        elif medication=='suboxone':
            pattern = r"suboxone\s*([^a-zA-Z]*\d+(?:\.\d+)?)"
        
        matches = re.findall(pattern, string, re.IGNORECASE)
        if len(matches)==1:
            dose_string = matches[0]
            dose = float(re.findall(numeric_pattern, dose_string, re.IGNORECASE)[0])
            return dose
        else:
            return np.nan
        
    
class WRAT3(object):
    
    def __init__(self):
        '''
        '''
        self.blue_scoring = pd.read_csv('wrat_scoring_input-raw_output-scaled_version-blue.csv')

        self.tan_scoring = pd.read_csv('wrat_scoring_input-raw_output-scaled_version-tan.csv')

        
        self.age_ranges = [
            (0,16),
            (17,19),
            (20,24),
            (25,34),
            (35,44),
            (45,54),
            (55,64),
        ]

    def scale_raw_score(self, wrat_total, age, version, input_col='wrat_total'):

        # get age range
        age_range = None
        for a in self.age_ranges:
            if int(age)>=a[0] and int(age)<=a[1]:
                age_range = a
                break
        
        # convert age range to string and column name
        age_range = str(age_range[0])+'_'+str(age_range[1])
        col = 'age_'+age_range

        # get scoring table
        if version=='blue':
            df_scoring = self.blue_scoring.copy()
        elif version=='tan':
            df_scoring = self.tan_scoring.copy()

        # check that column exists
        if col not in df_scoring.columns:
            wrat_scaled = np.nan
        else:
            wrat_scaled = df_scoring.set_index(input_col).loc[wrat_total][col]

        return wrat_scaled

class WASI(object):

    def __init__(self, ):
        '''
        '''
        self.raw2tscore = pd.read_csv('wasi_scoring_input_raw_output-tscore.csv')
        self.tscore2scaled = pd.read_csv('wasi_scoring_input-tscore_output_scaled.csv')

        self.age_ranges = [
            # (0,16),
            (17,19),
            (20,24),
            (25,29),
            (30,34),
            (35,44),
            (45,54),
            (55,64),
            (65,69)
        ]
        
    def scale_raw_score(self, wasi_total, age, input_col='wasi_total', tscore_col='t_score', scaled_col='scaled_score'):

        # get age range
        age_range = None
        for a in self.age_ranges:
            if int(age)>=a[0] and int(age)<=a[1]:
                age_range = a
                break
        
        # convert age range to string and column name
        age_range = str(age_range[0])+'_'+str(age_range[1])
        col = 'age_'+age_range

        # check that column exists
        if col not in self.raw2tscore.columns:
            scaled_score = np.nan
            tscore = np.nan
        else:
            if not wasi_total in self.raw2tscore[input_col].values:
                tscore=np.nan
                scaled_score = np.nan
            else:
                tscore = self.raw2tscore.set_index(input_col).loc[wasi_total][col]
                if tscore in self.tscore2scaled[tscore_col].values:
                    scaled_score = self.tscore2scaled.set_index(tscore_col).loc[tscore][scaled_col]
                else:
                    scaled_score = np.nan
        if np.isnan(scaled_score):
            breakpoint()
        return scaled_score, tscore
        

class ScanNotes(object):
    ''' convert scan notes to bids 'scans.tsv' file for each subject
    '''
    def __init__(self, filename=None, layout=None, run_all=False):
        '''
        '''
        # directories
        #--------------------------------------------------------------------
        # self.narclab_dir = '/home/gregkronberg/mriprojects/gkronberg/more_movie/'
        self.project_dir = '/home/gregkronberg/mriprojects/gkronberg/more_movie/'
        # self.project_dir = 'C:\\Users\\Greg Kronberg\\Google Drive\\Work\\Research Projects\\Mount Sinai\\MORE\\bids_data_temp\\'
        # self.project_dir = 'D:\\bids_data_temp\\'
        if filename is None:
            self.scan_notes_filename = 'MORE_MRI_Session_Notes.tsv'
        else:
            self.scan_notes_filename = filename
        # get bids layout
        #-------------------------------------------------------------------
        # if layout is None:
        #     self.layout = BIDSLayout(self.project_dir, validate=False)
        # else:
        #     self.layout = layout
        # self.df_layout = self.layout.to_df(metadata=True, extension='nii.gz')

        # get original scan notes and rename columns (see rename_scan_notes_columns for map)
        #-------------------------------------------------------------------
        if '.tsv' in self.scan_notes_filename:
            sep='\t'
        elif '.csv' in self.scan_notes_filename:
            sep=','
        self.scan_notes = pd.read_csv(self.scan_notes_filename,sep=sep)
        self.scan_notes = self.rename_scan_notes_columns(scan_notes=self.scan_notes)
        self.scan_notes['subject'] = self.scan_notes['subject'].apply(lambda x:x.strip())
        self.scan_notes = self.get_ordinal_datetime(scans_df=self.scan_notes)

        if run_all:
            self.run_all_subjects()

    def get_session_name_from_date(self, scans_df, subject, date=None, time=None, ):
        ''' deprecate, see get_session_from_date
        '''
        # FIXME needs lots of work
        # allow for two sessions on the same day
        # if a subject has multiple scans on the same day use time to check if rescan
        scans_df = scans_df.set_index(['subject'])
        subject_df = scans_df.loc[subject]
        subject_datetime = pd.to_datetime(subject_df['date']+' '+subject_df['time'])
        subject_datetime_rescan=np.nan
        if not pd.isna(subject_df['date_rescan']):
            if pd.isna(subject_df['time_rescan']):
                subject_df['time_rescan']='00:00:00'
            subject_datetime_rescan = pd.to_datetime(subject_df['date_rescan']+' '+subject_df['time_rescan'])

        return subject_datetime, subject_datetime_rescan

    @classmethod
    def get_session_from_date(cls, scan_df, subject, date, filter_rescans=True):
        '''
        Args:
            scan_df:pandas df:dataframe built from mri session notes
            subject:str:subject identifier 
            date:int:ordinal date
            filter_rescans:bool:if True, will return nan for the first session of a rescanned session
        Returns:
            session_name:str
        '''

        # if subject not in scan notes, skip
        if subject not in scan_df.subject.values:
            return np.nan
        # find all rows for the current subject
        sub_df = scan_df.set_index('subject').loc[subject]
        # if multiple rows (mri sessions), iterate rows and check for matching dates
        if isinstance(sub_df,pd.DataFrame):
            for _row_i, _row in sub_df.iterrows():
                # if the rating date is a rescan, keep it
                if date==_row.date_rescan_ordinal:
                    if not filter_rescans:
                        return str(_row.session)+'a'
                    else:
                        return str(_row.session)
                # if rating date is an original scan, check if there was a rescan
                elif date==_row.date_ordinal:
                    if not filter_rescans:
                        return str(_row.session)
                    else:
                        if pd.isna(_row.date_rescan_ordinal):
                            return str(_row.session)
                        else:
                            return np.nan
                else:
                    continue
            return np.nan
        # repeat cases of only only mri session
        else:
            if date==sub_df.date_rescan_ordinal:
                if not filter_rescans:
                    return str(sub_df.session)+'a'
                else:
                    return str(sub_df.session)
            elif date==sub_df.date_ordinal:
                if not filter_rescans:
                    return str(sub_df.session)
                else:
                    if ~pd.isna(sub_df.date_rescan_ordinal):
                        return str(sub_df.session)
                    else:
                        return np.nan
            else:
                return np.nan

    def rename_scan_notes_columns(self, scan_notes, **kwargs):
        '''
        ==Args==
        :scan_notes:df:original scan notes in dataframe format
        :scan_column_map:dict:optional:mapping between original and modified column names
        '''
        # map old to new columns to ease conversion
        #--------------------------------------------------------------------
        # default map
        self.scan_column_map={
        'Subject ID':'subject',
        'Group':'group',
        'Session (1/2)':'session',
        'Scan Date':'date',
        'Scan Time':'time',
        'Sequence Used (see Sheet 2)':'sequence_name',
        'Handedness (R/L)':'handedness',
        '1st Psychophys':'psychophys-1',
        'Localizer':'localizer',
        'Cue-reactivity Fieldmaps':'scan-fmap-cuereact',
        'Cue-reactivity':'scan-task-cuereact',
        'Stop-signal Fieldmaps':'scan-fmap-stopsignal',
        'Stop-signal':'scan-task-stopsignal',
        'Card-guessing':'scan-task-cardguess',
        '2nd Psychophys':'psychophys-2',
        'Movie':'scan-task-movie',
        'T1':'scan-anat-T1w',
        'T2':'scan-anat-T2w',
        'DWI':'scan-dwi',
        'Other Notes':'notes_other',
        'Rescan Ses? Y/N':'rescan',
        'Rescan Date':'date_rescan',
        'Rescan Time':'time_rescan',
        'Rescan Sequence':'rescan_sequence',
        'Rescan Notes':'notes_rescan',
        'Diffusion':'notes_diffusion'
        }
        # check args
        if 'scan_column_map' in kwargs:
            self.scan_column_map = kwargs['scan_column_map']
        # change column names and set datatype to string
        #--------------------------------------------------------------------
        new_columns = []
        for _col in scan_notes.columns:
            if _col in self.scan_column_map.keys():
                new_columns.append(self.scan_column_map[_col])
            else:
                new_columns.append(_col)
        scan_notes.columns=new_columns
        # scan_notes = scan_notes.astype(str)

        return scan_notes

    def run_all_subjects(self, qc_exclude_function='qc_exclude_rescan', max_subjects=None, **kwargs):
        ''' create scans.tsv for all subjects
        ==Args==
        :qc_exclude_function:str:points to function that uses a heuristic to exclude scans
        :max_subjects:int,None:maximum number of subjects to run
        '''
        # get subjects to run
        #--------------------------------------------------------------------
        if 'subjects' in kwargs:
            subjects=kwargs['subjects']
        elif 'layout_kwargs' in kwargs:
            layout_kwargs = kwargs['layout_kwargs']
            subjects = self.layout.get_subjects(**layout_kwargs)
        else:
            subjects = self.layout.get_subjects()

        # iterate subjects create scans.tsv and apply exclusion criteria to 'qc_exclude' column
        #--------------------------------------------------------------------
        for i, subject in enumerate(subjects):
            # check maximum number of subjects
            if max_subjects is None or max_subjects>i:
                # use layout and scan_notes to get subject-specific scans.tsv dataframe (this saves the df as sans.tsv)
                subject_df = self.subject_to_scans_tsv(subject)
                # check for quality control exclusion function
                if qc_exclude_function is not None and hasattr(self, qc_exclude_function):
                    exclude_func = getattr(self, qc_exclude_function)
                    # fill the qc_exclude column and save
                    subject_df = exclude_func(subject_df)
                else:
                    print('Exclusion function not found. Copying all data by default')
                
    def subject_to_scans_tsv(self, subject, **kwargs):
        ''' create subject level scans.tsv from spreadsheet of scan notes
        ==Args==
        :subject:str:subject to run
        '''
        # get list of scan entities
        scans = self.layout.get(subject=subject, extension='nii.gz')
        # dataframe for scans.tsv
        df=pd.DataFrame()
        df.index.name='filename'
        # iterate scans for the current subject
        for scan in scans:
            # get standard info: path, subject, session, datatype, suffix, run, direction
            #--------------------------------------------------------------- 
            # temporary dictionary (keys will become df columns)
            temp={}
            temp['path'] = scan.path
            temp['session'] = scan.entities['session']
            # check if session is a rescan by checking for letter in session name
            if 'run' in scan.entities:
                temp['run']=scan.entities['run']
            if 'direction' in scan.entities:
                temp['direction']=scan.entities['direction']
            temp['subject'] = scan.entities['subject']
            temp['suffix'] = scan.entities['suffix']
            temp['datatype'] = scan.entities['datatype']

            # check if the current scan is a rescan (indicated by a letter in the session number)
            #--------------------------------------------------------------
            rescan = len(re.findall('[a-z]',temp['session']))>0
            temp['is_rescan']=rescan

            # get session number and letter from session name
            #---------------------------------------------------------------
            if rescan:
                temp['session_number'] = re.findall('[0-9]',temp['session'])[0]
                temp['rescan_letter'] = re.findall('[a-z]',temp['session'])[0]
            else:
                temp['session_number'] = str(temp['session'])
            
            # copy notes that correspond to the current scan from original scan notes (rescans do not have notes for individual scans. instead there is a single notes_rescan column)
            #----------------------------------------------------------------
            notes=None
            # functional 
            #------------
            if temp['datatype'] == 'func':
                temp['task'] = scan.entities['task']
                if not rescan and hasattr(self,'scan_notes') and (temp['subject'],temp['session']) in self.scan_notes.set_index(['subject','session']).index:
                    notes = self.scan_notes.set_index(['subject','session']).loc[(temp['subject'],temp['session'])]['scan-task-'+temp['task']]
            # fieldmaps
            #------------
            if temp['datatype'] =='fmap':
                # get intendedfor files and task names
                #--------------------------------------
                temp['intendedfor'] = scan.entities['IntendedFor']
                if len(temp['intendedfor'])>0:
                    # intendedfor tasks
                    fmap_tasks = list(set([_val.split('task-')[-1].split('_')[0] for _val in temp['intendedfor'] if 'task-' in _val]))
                    temp['intendedfor_tasks'] = fmap_tasks
                    # fieldmap scan notes (rare)
                    if not rescan and hasattr(self,'scan_notes') and (temp['subject'],temp['session']) in self.scan_notes.set_index(['subject','session']).index:
                        for _col in self.scan_notes.columns:
                            if 'fmap' in _col and any([True for _task in fmap_tasks if _task in _col]):
                                notes = self.scan_notes.set_index(['subject','session']).loc[(temp['subject'],temp['session'])][_col]
            # anatomical
            #-------------
            if temp['datatype']=='anat':
                anat_type = scan.entities['suffix']
                if not rescan and hasattr(self,'scan_notes') and (temp['subject'],temp['session']) in self.scan_notes.set_index(['subject','session']).index:
                    notes = self.scan_notes.set_index(['subject','session']).loc[(temp['subject'],temp['session'])]['scan-anat-'+anat_type]
            # diffusion
            #-----------
            if temp['datatype']=='dwi':
                if not rescan and hasattr(self,'scan_notes') and (temp['subject'],temp['session']) in self.scan_notes.set_index(['subject','session']).index:
                    notes = self.scan_notes.set_index(['subject','session']).loc[(temp['subject'],temp['session'])]['scan-dwi']

            # copy notes to dict
            temp['scan_notes']=notes

            # add all scan info to dataframe
            #----------------------------------------------------------------
            for _key, _val in temp.items():
                if _key not in df:
                    df[_key]=None
                df.at[scan.filename, _key]=_val

            # copy remaining columns from original scan notes
            #----------------------------------------------------------------
            for _col in self.scan_notes.columns:
                # scan assumed to labeled with 'scan-'
                if 'scan-' not in _col and 'subject' not in _col and 'session'not in _col:
                    if _col not in df:
                        df[_col]=None
                    if hasattr(self,'scan_notes') and (temp['subject'],temp['session_number']) in self.scan_notes.set_index(['subject','session']).index:
                        df.at[scan.filename, _col] = self.scan_notes.set_index(['subject','session']).loc[(temp['subject'],temp['session_number'])][_col]
            # replace nan with 'None' (useful for checking equivalence between cells)
            df[df.isnull()]=None
            # get subject directory and save
            subject_dir = self.project_dir+'sub-'+subject+'/'
            filename = 'sub-'+subject+'_scans.tsv'
            df.to_csv(subject_dir+filename, sep='\t', index=True)

        return df

    def qc_exclude_rescan(self, df_scans, save=True):
        ''' exclude scans if they were redone (subject was brought back for rescan)
        ==Args==
        :df_scans:df:subject level scans.tsv file loaded as df
        :save:bool:save updated scans.tsv with exclusion column
        '''
        # column and value to mark excluded scans
        exclude_col = 'qc_exclude'
        exclude_val=1
        # column that links a scan to its rescan
        df_scans['rescan_files']=None
        # column that marks a file to be excluded
        df_scans[exclude_col]=None
        # iterate subject scans
        for filename in df_scans.index:
            # dont exclude rescans
            if df_scans.loc[filename].is_rescan:
                continue
            # get subject and session info
            subject = df_scans.loc[filename]['subject']
            session_number = df_scans.loc[filename]['session_number']
            # info needed to check for a rescan
            #----------------------------------------------------------------
            datatype=None
            task=None
            suffix=None
            direction=None
            datatype = df_scans.loc[filename]['datatype']
            if 'task' in df_scans.loc[filename]:
                task = df_scans.loc[filename]['task']
            if 'suffix' in df_scans.loc[filename]:
                suffix = df_scans.loc[filename]['suffix']
            if 'direction' in df_scans.loc[filename]:
                direction = df_scans.loc[filename]['direction']
            # match rescans based on the follwing columns
            #--------------------------------------------
            match_columns = ['session_number','datatype','task','suffix','direction']

            # get all matching filenames (there should be at least one for the current scan itself)
            #----------------------------------------------------------------
            df_temp = df_scans.reset_index().set_index(match_columns)
            matching_filenames = df_temp.loc[(session_number,datatype,task,suffix,direction)].filename
            
            # if there is more than one match, find rescans
            #----------------------------------------------------------------
            if len(matching_filenames)>1:
                # parameters match, but different filename, is a rescan, if fieldmap has the same intendedfor task types
                rescan_files = [_file for _file in matching_filenames if _file != filename and df_scans.loc[_file].is_rescan and df_scans.loc[_file].intendedfor_tasks==df_scans.loc[filename].intendedfor_tasks]
                if len(rescan_files)==0:
                    rescan_files=None
            else:
                rescan_files = None
            if rescan_files is not None:
                df_scans.at[filename, exclude_col]=exclude_val
            df_scans.at[filename, 'rescan_files']=rescan_files

        # replace nan with 'None'
        df_scans[df_scans.isnull()]=None
        
        # save
        #--------------------------------------------------------------------
        if save:
            # get subject directory and filename
            subject_dir = self.project_dir+'sub-'+subject+'/'
            filename = 'sub-'+subject+'_scans.tsv'
            # save
            df_scans.to_csv(subject_dir+filename, sep='\t', index=True)

        return df_scans

    def get_ordinal_datetime(self, scans_df):

        scans_df['date_ordinal'] = scans_df['date'].apply(lambda x: dateutil.parser.parse(x).date() if not pd.isna(x) else pd.to_datetime(x)).apply(lambda x: pd.datetime.toordinal(x) if not pd.isna(x) else x)
        scans_df['time_ordinal'] = scans_df['time'].apply(dateutil.parser.parse).apply(lambda x:x.time()).apply(lambda x:(60*60*x.hour + 60*x.minute + x.second)/(24*60*60))
        scans_df['date_rescan_ordinal'] = pd.to_datetime(scans_df['date_rescan']).apply(pd.datetime.toordinal)
        scans_df['date_rescan_ordinal'] = scans_df['date_rescan'].apply(lambda x: dateutil.parser.parse(x).date() if not pd.isna(x) else pd.to_datetime(x)).apply(lambda x: pd.datetime.toordinal(x) if not pd.isna(x) else x)
        scans_df['time_rescan_ordinal'] = scans_df['time_rescan'].apply(lambda x: dateutil.parser.parse(x).time() if not pd.isna(x) else pd.to_datetime(x)).apply(lambda x:(60*60*x.hour + 60*x.minute + x.second)/(24*60*60))
        return scans_df
