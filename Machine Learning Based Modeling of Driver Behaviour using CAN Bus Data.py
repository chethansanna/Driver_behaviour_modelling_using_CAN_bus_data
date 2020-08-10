# Import libraries

import pandas as pd
import numpy as np
import datetime
import time
import math

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
import pickle
import warnings
import re
import concurrent.futures
import random
import scipy.stats

import requests
import geopy.geocoders
from geopy.geocoders import Nominatim

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_validate

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.manifold import TSNE


def find_ground_truth_for_road_classifiction(df):
    """
    Get the ground truth for each position information of the dataset.
    First checks if there exists position information of a sample; if exists then check if the current position information...
    ...is same as the previous position, if so then the previously detected road type would be copied. This is done in order...
    ...to avoid sending frequent requests to the Open Street map
    
    """
    for i in range(len(df)):
        if (str(df['Position'][i]) != 'nan'):
            if (df['Position'][df.index[i]] == df['Position'][df.index[i-1]]):
                df.loc[df.index[i], 'Road_type'] = df.loc[df.index[i-1], 'Road_type']
            else:
                df_road_classification = ground_truth_road_classification(df, i)
    return df_road_classification
            
def ground_truth_road_classification(df, i):
    location = df['Position'][df_road_classification.index[i]] # Get the position information
    geolocator = Nominatim(user_agent="specify_your_app_name_here")
    location = geolocator.reverse("{}".format(location)) # Give location data to geolocator
    
    # Fine specific information from the address of that location
    try:
        road = location.raw['address']['road']
    except KeyError:
        try:
            road = location.raw['address']['village']
        except KeyError:
            try:
                road = location.raw['address']['industrial']
            except:
                pass
            
    state = location.raw['address']['state']
    try:
        postcode = location.raw['address']['postcode']
    except:
        postcode = location.raw['address']['country']
    try:
        # Pass those address information to nominatim() to get the road type
        url = "https://nominatim.openstreetmap.org/search?q={},{},{}&format=json".format(road,state,postcode) 
        r = requests.get(url)
        text = r.json()
        Road_type = text[0]['type']
        str(Road_type)
        df.loc[df.index[i], 'Road_type'] = Road_type
    except:
        pass
    
    return df

def load_road_classification_dataset(file_name):
    # Load dataset of road classification
    df = pd.read_csv(file_name,sep=';')

    # To convert the integer timestamp to isoformat
    for x in range(len(df)):
        df['Timestamp'][x] = datetime.datetime.fromtimestamp(df['Timestamp'][x]/1e3)

    """
    Conversion of each Identifier into columns with its corresponding value; and Timestamp as index. This step is specific to
    the dataset utilized. If the dataset has already got position information, then this step could be skipped
    
    """
    df = df.pivot_table(columns= 'Signals', index = 'Timestamp', values='Value', aggfunc = 'first',dropna = False)
    
    return df

def change_data_type(df):
    # In case of any change in the datatype of the features while loading
    columns = df.columns.drop('Road_type')

    df[columns] = df[columns].apply(pd.to_numeric)

    df = df[(~df_road_classification.isnull()).all(axis=1)]
    
    return df
    
def merge_road_calssification_labels(df):
    
    """
    
    Add or delete road types based on the data
    
    """
    
    df = df.replace(['primary', 'secondary', 'tertiary', 'unclassified', 'primary_link'], 
                                                'connecting_roads')
    df = df.replace(['motorway', 'trunk', 'motorway_link'], 'motorway')
    df = df.replace(['residential', 'industrial', 'administrative', 'service', 'hamlet', 'park', 
                                                     'living_street', 'postcode', 'neighbourhood', 'track', 'bridge', 'tram_stop', 
                                                     'pedestrian', 'proposed', 'attraction', 'urban', 'village', 
                                                     'recreation_ground', 'quarter'], 'residential')
    return df
    
def scale(df):
    
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(df[columns])
    df_scaled = pd.DataFrame(data = scaled_data, index = df.index, columns = columns)
    df_scaled = pd.concat([df_scaled, df['Road_type']],axis =1)
    df_scaled = df_scaled[(~df_scaled.isnull()).all(axis=1)]
    
    return df_scaled
    
def normalized(df):
    
    norm = MinMaxScaler()
    df_norm = norm.fit_transform(df.drop('Road_type', axis=1))
    df_norm = pd.DataFrame(data = df_norm, columns = columns, index = df.index)
    df_norm = pd.concat([df_norm, df['Road_type']],axis =1)
    df_norm = df_norm[(~df_norm.isnull()).all(axis=1)]
    
    return df_norm

def moving_mean_SD(data, window_period, sp):
    shift_period = sp * window_period
    i=0
    j=0

    columns  = data.columns
    #creating an empty dataset to append all the moving mean values
    df_mean = pd.DataFrame(index = data.index, columns = Xl1_signals)
    df_sd = pd.DataFrame(index = data.index, columns = Xl1_signals)
    df_temp_mean = pd.DataFrame(index = data.index, columns = columns)
    df_temp_sd = pd.DataFrame(index = data.index, columns = columns)

    while True:
        if i < len(data.index):
            if j < len(data.index):
                if ((abs((data.index[j] - data.index[i]).total_seconds())) >= window_period):
                    df_mean.iloc[math.ceil((j+i)/2)] = data[i:j].mean()
                    df_sd.iloc[math.ceil((j+i)/2)] = data[i:j].std(ddof = 1)
                    df_temp_mean.iloc[math.ceil((j+i)/2)] = data[i:j].mean()
                    df_temp_sd.iloc[math.ceil((j+i)/2)] = data[i:j].std(ddof = 1)
                    i += math.ceil((j-i)/(window_period/shift_period))
                else:
                    j+=1
            else:
                i=0
                j=0
                break
    df_mean['Road_type'] = data['Road_type']
    df_sd['Road_type'] = data['Road_type']
    df_temp_mean = df_temp_mean.add_suffix('_Mean')
    df_temp_sd = df_temp_sd.add_suffix('_SD')
    df_mean_sd = pd.concat([df_temp_mean, df_temp_sd], axis = 1)
    df_mean_sd['Road_type'] = data['Road_type']

    # To keep only the rows which has values
    df_mean = df_mean[(~df_mean.isnull()).all(axis=1)]
    df_sd = df_sd[(~df_sd.isnull()).all(axis=1)]
    df_mean_sd = df_mean_sd[(~df_mean_sd.isnull()).all(axis=1)]
    return df_mean, df_sd, df_mean_sd

def transition_signals(data, time):
    data.index = pd.to_datetime(data.index, infer_datetime_format=True)

    labels = data['Road_type'].unique()

    df_trans = pd.DataFrame(columns = data.columns)
    # Initialization
    i = 0
    j = 1
    while j < len(data.index):
        if data['Road_type'][i] != data['Road_type'][j]:
            icounter = 0
            counter_val = 0
            while i-icounter > 0:
                if (abs((data.index[i] - data.index[i-icounter]).total_seconds())) < time: 
                    icounter += 1
                    counter_val = i-icounter
                else:
                    icounter = i

            df_trans = pd.concat([df_trans, data.loc[data.index[counter_val:i]]])
            jcounter = 0
            counter_val = 0
            while  j+jcounter < len(data.index):
                if (abs((data.index[j] - data.index[j+jcounter]).total_seconds())) < time: 
                    jcounter += 1
                    counter_val = j+jcounter
                else:
                    jcounter = len(data.index)

            df_trans = pd.concat([df_trans, data.loc[data.index[j:counter_val]]]) 
            i = counter_val+1
            j = i+1
        else:
            i+=1
            j+=1

    df_trans = df_trans[(~df_trans.isnull()).all(axis=1)] # to keep only the rows without NaN values 
    df_trans.drop_duplicates(inplace = True) # Drop duplicates
    return df_trans

def road_classification_model_sampled_data(data, Classifiers):

    labels = data['Road_type'].unique()
    all_signals = data.columns.drop('Road_type')
    all_signals_but_RPS = data.columns.drop(['RPS', 'Road_type']) # RPS signals is removed since it is perfectly correlated with speed
    
    for features in [all_signals, all_signals_but_RPS]:
        if features == all_signals:
            feature_name = 'all_signals'
        elif features == all_signals_but_RPS:
            feature_name = 'all_signals_but_RPS'

        X = data[features]
        y = data['Road_type']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        for model in Classifiers:
            if model == svc:
                model_name = 'SVC'
            elif model == decisiontree:
                model_name = 'Decision_Tree'
            elif model == randomforest:
                model_name = 'Random_Forest'

            globals()['validation_f1_weighted_%s' %(feature_name)] = []
            globals()['validation_accuracy_%s' %(feature_name)] = []
            globals()['test_f1_weighted_%s' %(feature_name)] = []
            globals()['test_accuracy_%s' %(feature_name)] = []

            warnings.filterwarnings('ignore')
            scores = cross_validate(model, X_train, y_train, cv = 5, scoring = ['f1_weighted', 'balanced_accuracy'])
            globals()['validation_f1_weighted_%s' %(feature_name)] = scores['test_f1_weighted']
            globals()['validation_accuracy_%s' %(feature_name)] = scores['test_balanced_accuracy']
            model.fit(X_train, y_train)

            y_predict = model.predict(X_test)
            warnings.filterwarnings('ignore')
            globals()['test_f1_weighted_%s' %(feature_name)].append(f1_score(y_test, y_predict,average='weighted'))
            globals()['test_accuracy_%s' %(feature_name)].append(accuracy_score(y_test, y_predict))
            report = classification_report(y_test, y_predict, digits=3, output_dict=True)
            print(model_name, '\n', report)
        
    # Visualization of the validation and test set results
    fig = go.Figure()
    for result_type in ['validation', 'test']:
        for score_method in ['f1_weighted', 'accuracy']:
            if score_method == 'f1_weighted':
                score_name = 'F1 score'
            elif score_method == 'accuracy':
                score_name = 'Accuracy'

            i = 0
            fig.data = []
            fig.layout = {}
            for feature_name in ['all_signals', 'all_signals_but_RPS']:
                for alg_name in ['SVC', 'Decision_Tree', 'Random_Forest']:
                    score_list = '%s_%s_%s_%s' %(result_type, score_method, feature_name, alg_name)
                    std = np.std(globals()[score_list],ddof=1)
                    ci_list = (1.960 * std)/math.sqrt(len(globals()[score_list]))
                    hexcode = ['#A1D7DA', '#6E1D0C', '#F0E7B1', '#F99417', '#D16613', '#0011E7', '#AEB20C', '#D4B754', '#2B335E', 
                               '#FFCBAF', '#E10F2A',' #E448E7', '#207EDF', '#FE326C', '#367543', '#E4D630','#251B0B', '#00C132']

                    if result_type == 'validation':                        
                        fig.add_trace(go.Bar(
                            x=[re.sub('_', '', alg_name)],
                            y=[np.mean(globals()[score_list])],
                            name= re.sub('_', ' ', '%s'%(feature_name)),
                            error_y=dict(type='data', array=[ci_list]),
                            marker_color = hexcode[i],
                            width = 0.12
                        ))
                    elif result_type == 'test':
                        fig.add_trace(go.Bar(
                            x=[re.sub('_', '', alg_name)],
                            y=[np.mean(globals()[score_list])],
                            name= re.sub('_', ' ', '%s'%(feature_name)),
                            marker_color = hexcode[i],
                            text=np.mean(globals()[score_list]),
                            width = 0.12
                        ))
                        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', textfont_size=30)
                    i+=1
            fig.update_yaxes(range=(0, 1))
            fig.update_layout(
                barmode='group',
                title=re.sub('_', ' ','%s_%s' %(result_type, score_method)),
                xaxis_title="Algorithms",
                yaxis_title=score_name,
                font=dict(
                    family="Times New Roman",
                    size=14,
                    color="black"
                )
            )

            fig.show()
    return model

def road_classification_model_moving_mean_and_or_sd(option, type_of_data, Classifiers):
    for window_period in [5, 10, 15, 20, 25, 30]:
        for shift_period in [1/4 * window_period, 1/2 * window_period, 3/4 * window_period, 1 * window_period]:
            
            # Applying moving window technique on the input dataset for different window and shift period
            if option == 1:
                data, _, _ = moving_mean_SD(type_of_data, window_period, shift_period)
                all_signals_but_RPS = data.columns.drop(['RPS', 'Road_type']) # RPS signals is removed since it is perfectly correlated with speed
            elif option == 2:
                _, data, _ = moving_mean_SD(type_of_data, window_period, shift_period)
                all_signals_but_RPS = data.columns.drop(['RPS', 'Road_type']) # RPS signals is removed since it is perfectly correlated with speed
            elif option == 3:
                _, _, data = moving_mean_SD(type_of_data, window_period, shift_period)
                all_signals_but_RPS = data.columns.drop(['RPS_Mean', 'RPS_SD', 'Road_type']) # RPS signals is removed since it is perfectly correlated with speed
            
            labels = data['Road_type'].unique()
            all_signals = data.columns.drop('Road_type')
            
            for features in [all_signals, all_signals_but_RPS]:
                if features == all_signals:
                    feature_name = 'all_signals'
                elif features == all_signals_but_RPS:
                    feature_name = 'all_signals_but_RPS'

                X = data[features]
                y = data['Road_type']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                for model in Classifiers:
                    if model == svc:
                        model_name = 'SVC'
                    elif model == decisiontree:
                        model_name = 'Decision_Tree'
                    elif model == randomforest:
                        model_name = 'Random_Forest'
                    
                    # Assigning variables for each window and shift period; feature; model
                    globals()['validation_f1_weighted_%i_%.6f_%s_%s' %(window_period, shift_period, feature_name, model_name)] = []
                    globals()['validation_accuracy_%i_%.6f_%s_%s' %(window_period, shift_period, feature_name, model_name)] = []
                    globals()['test_f1_weighted_%i_%.6f_%s_%s' %(window_period, shift_period, feature_name, model_name)] = []
                    globals()['test_accuracy_%i_%.6f_%s_%s' %(window_period, shift_period, feature_name, model_name)] = []
                    
                    # Cross-validation method was used. Hence model is tested against validation set
                    filterwarnings('ignore')
                    scores = cross_validate(model, X_train, y_train, cv = 5, scoring = ['f1_weighted', 'balanced_accuracy'], n_jobs=-1)
                    globals()['validation_f1_weighted_%i_%.6f_%s_%s' %(window_period, shift_period, feature_name, model_name)] = scores['test_f1_weighted']
                    globals()['validation_accuracy_%i_%.6f_%s_%s' %(window_period, shift_period, feature_name, model_name)] = scores['test_balanced_accuracy']
                    model.fit(X_train, y_train)
                    
                    # Model is tested against test set
                    y_predict = model.predict(X_test)
                    warnings.filterwarnings('ignore')
                    globals()['test_f1_weighted_%i_%.6f_%s_%s' %(window_period, shift_period, feature_name, model_name)].append(f1_score(y_test, y_predict,average='weighted'))
                    globals()['test_accuracy_%i_%.6f_%s_%s' %(window_period, shift_period, feature_name, model_name)].append(accuracy_score(y_test, y_predict))
                    report = classification_report(y_test, y_predict, digits=3, output_dict=True)
                    print(report)
                    
    # Visualization of the validation and test set results
    fig = go.Figure()
    for feature_name in ['all_signals', 'all_signals_but_RPS']:                
        for model_name in ['SVC', 'Decision_Tree', 'Random_Forest']:
            for result_type in ['validation', 'test']:
                for score_method in ['f1_weighted', 'accuracy']:
                    if score_method == 'f1_weighted':
                        score_name = 'F1 score'
                    elif score_method == 'accuracy':
                        score_name = 'Accuracy'
                    i = 0
                    fig.data = []
                    fig.layout = {}
                    for window_period in [5, 10, 15, 20, 25, 30]:
                        scores=[]
                        ci = []
                        for shift_period in [1/4 * window_period, 1/2 * window_period, 3/4 * window_period, 1 * window_period]:
                            score_list = '%s_%s_%i_%.6f_%s_%s' %(result_type, score_method, window_period, shift_period, feature_name, model_name)

                            hexcode = ['#A1D7DA', '#6E1D0C', '#F0E7B1', '#F99417', '#D16613', '#0011E7', '#AEB20C', '#D4B754', '#2B335E', 
                                       '#FFCBAF', '#E10F2A',' #E448E7', '#207EDF', '#FE326C', '#367543', '#E4D630','#251B0B', '#00C132']
                            scores.append(np.mean(globals()[score_list]))
                            std = np.std(globals()[score_list],ddof=1)
                            ci.append((1.960 * std)/math.sqrt(len(globals()[score_list])))
                        if result_type == 'validation':
                            fig.add_trace(go.Bar(
                                x=['1/4', '1/2', '3/4', '1'],
                                y=scores,
                                name= 'window period: %i'.strip('_') %(window_period),
                                error_y=dict(type='data', array=ci), 
                                marker_color = hexcode[i+4],
                            ))
                        elif result_type == 'test':
                            fig.add_trace(go.Bar(
                                x=['1/4', '1/2', '3/4', '1'],
                                y=scores,
                                name= 'window period: %i'.strip('_') %(window_period), 
                                text=scores,
                                marker_color = hexcode[i+4],
                            ))
                            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        i+=1


                    fig.update_yaxes(range=(0, 1))
                    fig.update_layout(
                        barmode='group',
                        bargroupgap=0.15,
                        title="%s %s of %s using %s".strip('_') %(result_type, score_method, feature_name, model_name),
                        xaxis_title="Shift period",
                        yaxis_title=score_name,
                        font=dict(
                            family="Times New Roman",
                            size=14,
                            color="black"
                        )
                    )

                    fig.show()
    return model

def road_classification_model_transition_signals(df_trans, data):
 
    labels = data['Road_type'].unique()
    all_signals = data.columns.drop('Road_type')
    all_signals_but_RPS = data.columns.drop(['RPS', 'Road_type']) # RPS signals is removed since it is perfectly correlated with speed
    
    for features in [all_signals, all_signals_but_RPS]:
        if features == all_signals:
            feature_name = 'all_signals'
        elif features == all_signals_but_RPS:
            feature_name = 'all_signals_but_RPS'
            
        X_train = df_trans[features]
        y_train = df_trans['Road_type']
        X_test = data[features]
        y_test = data['Road_type']

        for model in Classifiers:
            if model == svc:
                model_name = 'SVC'
            elif model == decisiontree:
                model_name = 'Decision_Tree'
            elif model == randomforest:
                model_name = 'Random_Forest'

            globals()['test_f1_weighted_%s_%s' %(feature_name, model_name)] = []
            globals()['test_accuracy_%s_%s' %(feature_name, model_name)] = []

            warnings.filterwarnings('ignore')

            model.fit(X_train,y_train)
            y_predict = model.predict(X_test)
            globals()['test_f1_weighted_%s_%s' %(feature_name, model_name)].append(f1_score(y_test, y_predict,average='weighted'))
            globals()['test_accuracy_%s_%s' %(feature_name, model_name)].append(accuracy_score(y_test, y_predict))
            report = classification_report(y_test, y_predict, digits=3, output_dict=True)
            
    fig = go.Figure()
    result_type = 'test'
    for score_method in ['f1_weighted', 'accuracy']:
        if score_method == 'f1_weighted':
            score_name = 'F1 score'
        elif score_method == 'accuracy':
            score_name = 'Accuracy'
        i = 0
        fig.data = []
        fig.layout = {}
        for feature_name in ['all_signals', 'all_signals_but_RPS']:
            for alg_name in ['SVC', 'Decision_Tree', 'Random_Forest']:
                score_list = 'Trans_%s_%s_%s_%s_%s_%s' %(result_type, score_method, stand, road_types, feature_name, alg_name)
                std = np.std(globals()[score_list],ddof=1)
                ci_list = (1.960 * std)/math.sqrt(len(globals()[score_list]))
                hexcode = ['#A1D7DA', '#6E1D0C', '#F0E7B1', '#F99417', '#D16613', '#0011E7', '#AEB20C', '#D4B754', '#2B335E', 
                           '#FFCBAF', '#E10F2A',' #E448E7', '#207EDF', '#FE326C', '#367543', '#E4D630','#251B0B', '#00C132']

                if result_type == 'test':
                    fig.add_trace(go.Bar(
                        x=[re.sub('_', '', alg_name)],
                        y=[np.mean(globals()[score_list])],
                        name= re.sub('_', ' ', '%s'%(feature_name)),
                        marker_color = hexcode[i],
                        text=np.mean(globals()[score_list]),
                        width = 0.12
                    ))
                    i+=1
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', textfont_size=30)

        fig.update_yaxes(range=(0, 1))
        fig.update_layout(
            barmode='group',
            title=re.sub('_', ' ','Transition_%s_%s_%s' %(result_type, score_method)),
            xaxis_title="Algorithms",
            yaxis_title=score_name,
            font=dict(
                family="Times New Roman",
                size=14,
                color="black"
            )
        )

        fig.show()
    return model

"""
Different funtions for each GMM (Gaussian Mixture Model) to leverage the use of parallel processing
"""
def GMM_full(X):
    silhouette_score_GMM_full = []
    for n_components in np.arange(2, 5):
        GMM = GaussianMixture(n_components=n_components, covariance_type='full', max_iter = 150)
        globals()['cluster_labels_GMM_full_%i'%n_components] = GMM.fit_predict(X)
        silhouette_score_GMM_full.append(silhouette_score(X, globals()['cluster_labels_GMM_full_%i'%n_components]))
    return ['silhouette_score_GMM_full', silhouette_score_GMM_full, 'GMM_full_n_components = 2', cluster_labels_GMM_full_2, 'GMM_full_n_components = 3',
            cluster_labels_GMM_full_3, 'GMM_full_n_components = 4', cluster_labels_GMM_full_4]

def GMM_tied(X):
    silhouette_score_GMM_tied = []
    for n_components in np.arange(2, 5):
        GMM = GaussianMixture(n_components=n_components, covariance_type='tied', max_iter = 150)
        globals()['cluster_labels_GMM_tied_%i'%n_components] = GMM.fit_predict(X)
        silhouette_score_GMM_tied.append(silhouette_score(X, globals()['cluster_labels_GMM_tied_%i'%n_components]))
    return ['silhouette_score_GMM_tied', silhouette_score_GMM_tied, 'GMM_tied_n_components = 2', cluster_labels_GMM_tied_2, 'GMM_tied_n_components = 3',
            cluster_labels_GMM_tied_3, 'GMM_tied_n_components = 4', cluster_labels_GMM_tied_4]

def GMM_diag(X):
    silhouette_score_GMM_diag = []
    for n_components in np.arange(2, 5):
        GMM = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter = 150)
        globals()['cluster_labels_GMM_diag_%i'%n_components] = GMM.fit_predict(X)
        silhouette_score_GMM_diag.append(silhouette_score(X, globals()['cluster_labels_GMM_diag_%i'%n_components]))
    return ['silhouette_score_GMM_diag', silhouette_score_GMM_diag, 'GMM_diag_n_components = 2', cluster_labels_GMM_diag_2, 'GMM_diag_n_components = 3',
            cluster_labels_GMM_diag_3, 'GMM_diag_n_components = 4', cluster_labels_GMM_diag_4]

def GMM_spherical(X):
    silhouette_score_GMM_spherical = []
    for n_components in np.arange(2, 5):
        GMM = GaussianMixture(n_components=n_components, covariance_type='spherical', max_iter = 150)
        globals()['cluster_labels_GMM_spherical_%i'%n_components] = GMM.fit_predict(X)
        silhouette_score_GMM_spherical.append(silhouette_score(X, globals()['cluster_labels_GMM_spherical_%i'%n_components]))
    return ['silhouette_score_GMM_spherical', silhouette_score_GMM_spherical, 'GMM_spherical_n_components = 2', cluster_labels_GMM_spherical_2, 
            'GMM_spherical_n_components = 3', cluster_labels_GMM_spherical_3, 'GMM_spherical_n_components = 4', cluster_labels_GMM_spherical_4]

def Kmeans(X):
    silhouette_score_kmeans = []
    for n_clusters in np.arange(2, 5):
        Kmeans = KMeans(n_clusters=n_clusters, n_jobs = -1)
        globals()['cluster_labels_Kmeans_%i'%n_clusters] = Kmeans.fit_predict(X)
        silhouette_score_kmeans.append(silhouette_score(X, globals()['cluster_labels_Kmeans_%i'%n_clusters]))
    return ['silhouette_score_kmeans', silhouette_score_kmeans, 'KM_n_clusters = 2', cluster_labels_Kmeans_2, 'KM_n_clusters = 3',
            cluster_labels_Kmeans_3, 'KM_n_clusters = 4', cluster_labels_Kmeans_4]

def figure(road):
    fig = plt.figure(figsize=(15, 10))
    fig.title('Sihouette score of KMeans and Gaussian Mixture Model using %s data'%road, fontsize=16)

    plt.plot(np.arange(2, 5), silhouette_score_kmeans, linestyle='-', marker='o', label='Kmeans')
    plt.plot(np.arange(2, 5), silhouette_score_GMM_full, linestyle='-', marker='o', label='GMM_full')
    plt.plot(np.arange(2, 5), silhouette_score_GMM_tied, linestyle='-', marker='o', label='GMM_tied')
    plt.plot(np.arange(2, 5), silhouette_score_GMM_diag, linestyle='-', marker='o', label='GMM_diag')
    plt.plot(np.arange(2, 5), silhouette_score_GMM_spherical, linestyle='-', marker='o', label='GMM_spherical')
    plt.xlabel('Number of components (clusters)')
    plt.ylabel('Silhouette Score')  # The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
    plt.legend()
    plt.show()

def cluster_visualization(X, df, df_cluster_labels, road):
    # Dimensionality reduction
    n_components = 2  # The input features would be reduced to given number of dimension
    time_start = time()
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X.values) 
    print('t-SNE done! Time elapsed: {} seconds'.format(time() - time_start))

    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    df_data_and_labels = pd.concat([df, df_cluster_labels], axis=1)
    
    # Visualization of the clusters
    for n_clusters in np.arange(2, 5):
        for alg_name in ['KM_n_clusters = %i' % (n_clusters),
                         'GMM_full_n_components = %i' % (n_clusters),
                         'GMM_tied_n_components = %i' % (n_clusters),
                         'GMM_diag_n_components = %i' % (n_clusters),
                         'GMM_spherical_n_components = %i' % (n_clusters)]:
            plt.figure(figsize=(15, 10))
            sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue=alg_name,
                            palette=sns.color_palette("hls", n_colors=n_clusters), data=df_data_and_labels,
                            legend="full", alpha=0.3)
            plt.show()
    return df_data_and_labels

def LabellingBehaviour(df):
    # Label the behaviour of the driver based on the thresholds of lateral and longitudinal acceleration
    df['Behaviour'] = 'empty'

    df['Behaviour'].mask(df[(df['BR2_Querbeschl'] >= 1) | (df['BR2_Querbeschl'] <= -1) | (df['BR8_Laengsbeschl'] >= 3) |
                            (df['BR8_Laengsbeschl'] <= -3)]['Behaviour'] == 'empty', 'Aggressive', inplace=True)

    df['Behaviour'].mask(df[(df['BR2_Querbeschl'] < 1) | (df['BR2_Querbeschl'] > -1) | (df['BR8_Laengsbeschl'] < 3) |
                            (df['BR8_Laengsbeschl'] > -3)]['Behaviour'] == 'empty', 'Normal', inplace=True)

    return df

def F1_score_for_each_road_type(df_raw_data, df_scaled):
    n_clusters = 2
    df_ultimate_final = pd.DataFrame()
    for data, data_type in zip([df_raw_data, df_scaled], ['Raw data', 'Scaled']): 
        columns = pd.MultiIndex.from_tuples([('%s'%data_type, 'Normal'), ('%s'%data_type, 'Aggressive'), 
                                             ('%s'%data_type, 'Average'), ('%s'%data_type, 'Weighted average')])
        df_final_class_report = pd.DataFrame(columns = columns, index=['Kmeans', 'GMM-full', 'GMM-tied', 'GMM-diag', 'GMM-spherical'])
        f1_aggressive = []
        f1_normal = []
        macro_avg = []
        weighted_avg = []
        for alg_name in ['KM_n_clusters = %i'%n_clusters, 'GMM_full_n_components = %i'%n_clusters, 'GMM_tied_n_components = %i'%n_clusters, 
                     'GMM_diag_n_components = %i'%n_clusters, 'GMM_spherical_n_components = %i'%n_clusters]:

            for n in np.arange(n_clusters):
                data_normal = data[(data[alg_name] == n) & (data['Behaviour'] == 'Normal')]
                data_agg = data[(data[alg_name] == n) & (data['Behaviour'] == 'Aggressive')]

                if len(data_normal) > len(data_agg):        
                    data[alg_name].mask(data[alg_name] == n, 'Normal', inplace=True)
                elif len(data_normal) < len(data_agg):        
                    data[alg_name].mask(data[alg_name] == n, 'Aggressive', inplace=True)   
            y_true = data['Behaviour']
            y_pred = data[alg_name]
            df_class_report = classification_report(y_true,y_pred,output_dict=True)
            f1_aggressive.append(df_class_report['Aggressive']['f1-score'])
            f1_normal.append(df_class_report['Normal']['f1-score'])
            macro_avg.append(df_class_report['macro avg']['f1-score'])
            weighted_avg.append(df_class_report['weighted avg']['f1-score'])

        df_final_class_report[data_type, 'Aggressive'] = f1_aggressive
        df_final_class_report[data_type, 'Normal'] = f1_normal
        df_final_class_report[data_type, 'Average'] = macro_avg
        df_final_class_report[data_type, 'Weighted average'] = weighted_avg
        df_ultimate_final = pd.concat([df_ultimate_final, df_final_class_report], axis=1)
    display(df_ultimate_final)
    
    
 
 

if __name__ == '__main__':
    ################################################## Part 1 of the thesis: Road Classification ##################################################

    ################ Load the dataset for which road classification has to be carried out ###############

    df_road_classification = load_road_classification_dataset('file_name')
    df_road_classification.set_index('Timestamp', inplace=True)

    ############### Data exploration and data cleaning ###############

        # Fill all the NaN values in DataFrame with Preceding values, since the CAN signals were updated to the database only...
        # ...when there was a change in value.
    df_road_classification.fillna(method = 'ffill', inplace=True)

        # In case of any change in the datatype of the features while loading
    df_road_classification = change_data_type(df_road_classification)
        
        # Exploratory data analysis
    display(df_road_classification.corr())
    display(df_road_classification.describe())
            

    ############### Data pre-processing ###############

        # To get ground truth from the 'position' information
    df_road_classification = find_ground_truth_for_road_classifiction(df_road_classification)

        # In case of combining labels of the dataset
    df_road_classification_merged = merge_road_calssification_labels(df_road_classification)

        # select 'any one' of the dataset using which road classification has to be carried out
    dataset_to_be_standaradized = df_road_classification # In case of using the dataset with original labels
    dataset_to_be_standaradized = df_road_classification_merged # In case of using the dataset with 'merged' labels

    df_road_classification_scaled = scale(dataset_to_be_standaradized) 
    df_road_classification_norm = normalize(dataset_to_be_standaradized)


    ############### Creating artificial features ###############

        # Moving mean & SD. 
        """
        Argument 1 (Data): Pass either "df_road_classification, df_road_classification_merged, df_road_classification_scaled or 
        df_road_classification_norm"
        Argument 2 (window_period):  could be either "5, 10, 15, 20, 25, or 30"
        Arugment 3 (shift_period): could be either "1/4, 1/2, 3/4 or 1"
        """
    df_mean, df_sd, df_mean_sd = moving_mean_SD(Argument 1, Argument 2, Argument 3) 
        
        # Transition signals (Only signals recorded for a particular time period before and after the road type change is used 
        # for training the model, however tested against entire dataset)
        """
        Argument 1 (Data): Pass either "df_road_classification, df_road_classification_merged, df_road_classification_scaled or 
        df_road_classification_norm"
        Argument 2 (time): Time period for which data has to be collected from each road during road transition
        """
    df_trans = transition_signals(Argument 1, Argument 2)

    ############### Implementation ###############

        # Create object for each algorithm
    svc = LinearSVC(class_weight = 'balanced', dual = False)
    decisiontree = DecisionTreeClassifier(class_weight='balanced', criterion= 'entropy')
    randomforest = RandomForestClassifier(n_estimators=150, class_weight='balanced')
    Classifiers = [randomforest, decisiontree, svc]
        
        # Method 1 - Just with sampled data
        """
        Argument 1: Data for which road type has to be identified. Pass either "df_road_classification, 
        df_road_classification_merged, df_road_classification_scaled or df_road_classification_norm"
        """
        
    model_sampled_data = road_classification_model_sampled_data(Argument 1, Classifiers)
        
        # Method 2 - Moving mean or Moving SD or both Moving mean and Movind SD
        """
        Argument 1 (method): Send one of the options below as arguments 
            1 - for Moving mean
            2 - for Moving SD
            3 - for both Moving mean and Movind SD
        Argument 2 (data): Type of data on which the moving window technique to be applied. Pass either "df_road_classification, 
        df_road_classification_merged, df_road_classification_scaled or df_road_classification_norm"
        
        """
    model_moving_mean_and_or_sd = road_classification_model_moving_mean_and_or_sd(Argument 1, Argument 2, Classifiers)

        # Method 3 - Using transition signals
        """
        Argument 1 (data): Type of data on which the model trained on transition signals to be applied. 
        Pass either "df_road_classification, df_road_classification_merged, df_road_classification_scaled or 
        df_road_classification_norm"
        """
    model_transition_signals = road_classification_model_transition_signals(df_trans, Argument 1) 
    
    
    ################################################## Part 2 of the thesis: Driver behaviour clustering ##################################################
        # Load the dataset for which driver behaviour clustering has to be carried out 
    df_driver_behaviour = pd.read_csv('file_name.csv')
    df_driver_behaviour.set_index('Timestamp', inplace=True)
    
        # Apply the previously trained & tested road classification model on the datasets used for driver behaviour clustering
    df_driver_behaviour['Road_type'] = model_sampled_data.predict(df_driver_behaviour)
    
        # Lable the samples with either 'normal' or 'aggressive' behaviour based on the thresholds of lateral and longitudinal acceleration
    df_driver_behaviour = LabellingBehaviour(df_driver_behaviour)
    
        # Collect all the aggressive data based on the thresholds of lateral and longitudinal acceleration from all the driver datasets
    df_aggressive_signals = pd.read_csv('file_name.csv')
    df_aggressive_signals.set_index('Timestamp', inplace=True)
    
        # Add the aggressive signals to each driver dataset
    df_final_driver_behaviour = pd.concat([df_final_driver_behaviour, df_aggressive_signals], axis=0)
    
        # Input features used for clustering 
    signals_unscaled = ['speed', 'rps', 'longitudinal_acceleration' 'gas_pedal_pressure', 'lateral_acceleration', 'yaw_rate', 'brake_pressure', 'gear']
    signals_scaled = ['BR1_Rad_kmh_scaled', 'BR2_Querbeschl_scaled', 'BR2_mi_Radgeschw_scaled', 'BR5_Bremsdruck_scaled',
                      'BR5_Giergeschw_scaled', 'BR8_Laengsbeschl_scaled', 'MO1_Pedalwert_scaled', 'GE2_akt_Gang_scaled']

        # Scaling of the input features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_final_driver_behaviour[signals_unscaled])
    df_scaled_final_driver_behaviour = pd.DataFrame(data=scaled_data, index=df_final_driver_behaviour.index, columns=signals_scaled)
    df_scaled_final_driver_behaviour = pd.concat([df_final_driver_behaviour, df_scaled_final_driver_behaviour], axis=1)
    df_scaled_final_driver_behaviour = df_scaled_final_driver_behaviour[(~df_scaled_final_driver_behaviour.isnull()).all(axis=1)]

        # Using raw_data as well as scaled data to find driver behaviour for each road type and for overall data of each driver
    for road in ['residential', 'connecting_roads', 'overall']:
        for input_data in ['raw_data', 'scaled']:
            if input_data == 'raw_data':
                signals = signals_unscaled
                driver_behaviour_data = df_final_driver_behaviour.copy()
            elif input_data == 'scaled':
                signals = signals_scaled
                driver_behaviour_data = df_scaled_final_driver_behaviour.copy()

            if road == 'overall':
                X = driver_behaviour_data[signals]
            else:
                X = driver_behaviour_data[driver_behaviour_data['Road_type'] == road][signals]
                
            X = X.astype('float32')
            
            df_cluster_labels = pd.DataFrame(index=X.index)
            # add the functions of the algorithms used for clustering
            functions = [GMM_full, GMM_tied, GMM_diag, GMM_spherical, Kmeans]
            # For parallel processing of the algorithms using all the cores of the processor    
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(i, X) for i in functions]
                for f in concurrent.futures.as_completed(results):
                    for temp in np.arange(0, 2, 2):
                        globals()[f.result()[temp]] = f.result()[temp + 1]
                    for temp in np.arange(2, 8, 2):
                        df_cluster_labels[f.result()[temp]] = f.result()[temp + 1]

            # Plotting silhouette score 
            figure(road)

            # Cluster visualization in 2D
            if road == 'overall':
                globals()['df_data_and_labels_%s'%input_data] = cluster_visualization(X, driver_behaviour_data, df_cluster_labels, road)
            else:
                globals()['df_data_and_labels_%s'%input_data] = cluster_visualization(X, driver_behaviour_data[driver_behaviour_data['Road_type'] == road], df_cluster_labels, road)
                
            # 'df_data_and_labels_raw_data, df_data_and_labels_scaled' - This dataframe would have clustering results of all the algorithms along 
            # with the input data
        
        # To find the F1 score with true labels - obtained from thresholds; predicted labels - obtained by algorithms
        print('F1 score for %s data \n' %road)
        F1_score_for_each_road_type(df_data_and_labels_raw_data, df_data_and_labels_scaled)
