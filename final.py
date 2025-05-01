import os
import ast
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# ======= Section 1: Fake Account Detection Pipeline =======

def process_fake_data(fake_csv: str, real_csv: str) -> pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler
    from imblearn.over_sampling import RandomOverSampler
    fake_df = pd.read_csv(fake_csv)
    real_df = pd.read_csv(real_csv)
    merged = pd.concat([fake_df, real_df], ignore_index=True)
    merged.dropna(inplace=True)
    merged.drop_duplicates(inplace=True)
    num_cols = ['userFollowerCount', 'userFollowingCount', 'userBiographyLength', 'userMediaCount']
    scaler_f = MinMaxScaler()
    merged[num_cols] = scaler_f.fit_transform(merged[num_cols])
    X, y = merged.drop(columns=['isFake']), merged['isFake']
    X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X, y)
    df_bal = pd.DataFrame(X_res, columns=X.columns)
    df_bal['isFake'] = y_res
    select_feats = ['userHasProfilPic', 'userFollowingCount', 'usernameDigitCount', 'userIsPrivate', 'userFollowerCount', 'isFake']
    final_df = df_bal[select_feats]
    return final_df


def split_fake_data(df: pd.DataFrame):
    X = df.drop(columns=['isFake'])
    y = df['isFake']
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def train_fake_classical(X_tr, y_tr, X_te, y_te):
    models = {'logistic': LogisticRegression(max_iter=500), 'random_forest': RandomForestClassifier(random_state=42), 'svm': SVC()}
    reports = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        preds = mdl.predict(X_te)
        reports[name] = classification_report(y_te, preds, output_dict=True)
    return models, reports


def train_fake_mlp(X_tr, y_tr, X_te, y_te):
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=500, random_state=42)
    mlp.fit(X_tr, y_tr)
    preds = mlp.predict(X_te)
    report = classification_report(y_te, preds, output_dict=True)
    return mlp, report


def run_fake_pipeline(fake_csv: str, real_csv: str):
    df = process_fake_data(fake_csv, real_csv)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_fake_data(df)
    fake_models, fake_reports = train_fake_classical(X_tr, y_tr, X_te, y_te)
    mlp_model, mlp_rep = train_fake_mlp(X_tr, y_tr, X_te, y_te)
    fake_reports['mlp'] = mlp_rep
    fake_models['mlp'] = mlp_model
    fake_features = list(X_tr.columns)
    return fake_models, fake_reports, fake_features
# ======= Section 2: Automated Detection Pipeline =======
LIST_COLS = [
    'mediaLikeNumbers', 'mediaCommentNumbers', 'mediaCommentsAreDisabled',
    'mediaHashtagNumbers', 'mediaUploadTimes', 'mediaHasLocationInfo'
]

def process_auto_raw(df: pd.DataFrame) -> pd.DataFrame:
    # Parse stringified lists and compute stats
    for col in LIST_COLS:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else np.nan)
    df.dropna(inplace=True)
    for col in LIST_COLS:
        df[f'{col}_mean'] = df[col].apply(np.mean)
        df[f'{col}_std']  = df[col].apply(np.std)
        df[f'{col}_min']  = df[col].apply(np.min)
        df[f'{col}_max']  = df[col].apply(np.max)
    df.drop(columns=LIST_COLS, inplace=True)
    return df


def process_auto_data(auto_csv: str, non_auto_csv: str) -> pd.DataFrame:
    df_a = pd.read_csv(auto_csv)
    df_na = pd.read_csv(non_auto_csv)
    for df in (df_a, df_na):
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = process_auto_raw(df)
    merged = pd.concat([df_a, df_na], ignore_index=True).sample(frac=1, random_state=42)
    label = 'automatedBehaviour'
    feature_cols = [c for c in merged.columns if c != label]
    scaler_a = MinMaxScaler()
    merged[feature_cols] = scaler_a.fit_transform(merged[feature_cols])
    X_res, y_res = RandomOverSampler(random_state=42).fit_resample(merged[feature_cols], merged[label])
    df_bal = pd.DataFrame(X_res, columns=feature_cols)
    df_bal[label] = y_res
    return df_bal, feature_cols, scaler_a


def split_auto_data(df: pd.DataFrame):
    label = 'automatedBehaviour'
    X = df.drop(columns=[label])
    y = df[label]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def train_auto_models(X_tr, y_tr, X_te, y_te):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC()
    }
    reports = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        preds = mdl.predict(X_te)
        reports[name] = classification_report(y_te, preds, output_dict=True)
    return models, reports


# def run_auto_pipeline(auto_csv: str, non_auto_csv: str):
#     df_bal, auto_feats, auto_scaler = process_auto_data(auto_csv, non_auto_csv)
#     X_tr, X_val, X_te, y_tr, y_val, y_te = split_auto_data(df_bal)
#     models, reports = train_auto_models(X_tr, y_tr, X_te, y_te)
#     return models, reports, auto_feats, auto_scaler
def run_auto_pipeline(auto_csv: str, non_auto_csv: str):
    df_bal, auto_feats, auto_scaler = process_auto_data(auto_csv, non_auto_csv)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_auto_data(df_bal)
    models, reports = train_auto_models(X_tr, y_tr, X_te, y_te)
    return models, reports, auto_scaler, auto_feats 


# ======= Section 3: Load & Cache Pipelines =======
@st.cache_resource
def load_fake():
    return run_fake_pipeline('fakeAccountData.csv', 'realAccountData.csv')

@st.cache_resource
def load_auto():
    return run_auto_pipeline('automatedAccountData.csv', 'nonautomatedAccountData.csv')

fake_models, fake_reports, fake_feats = load_fake()
auto_models, auto_reports, auto_scaler, auto_feats = load_auto()

# ======= Section 4: Streamlit UI =======
st.sidebar.title('Navigation')
option = st.sidebar.radio('Choose a page', ['🏠 Home', '🧪 Fake Account Detection', '🤖 Bot Automated Detection', '📊 Fake Account Performance', '📊 Bot Automated Performance'])

if option == '🏠 Home':
    st.title('🤖 Social Media Account Analysis Toolkit')
    st.write('Choose from Fake or Automated detection, or view performance metrics.')

elif option == '🧪 Fake Account Detection':
    st.title('🧪 Fake Account Detection')
    upload = st.file_uploader('📁 Upload CSV for 🧪 Fake Detection', type=['csv'])
    if upload:
        df = pd.read_csv(upload)
        missing = [f for f in fake_feats if f not in df.columns]
        if missing:
            st.error(f'Missing columns: {missing}')
        else:
            df[fake_feats] = df[fake_feats].apply(lambda col: (col - col.min())/(col.max()-col.min()))
            preds = fake_models['random_forest'].predict(df[fake_feats])
            df['Prediction'] = ['Fake' if p==1 else 'Real' for p in preds]
            st.dataframe(df)
    else:
        st.subheader('🔍 Manual Input')
        inp = {f: st.number_input(f.replace('_',' ').title(), value=0.0) for f in fake_feats}
        if st.button('✅ Predict'):
            df2 = pd.DataFrame([inp])
            df2[fake_feats] = df2[fake_feats].apply(lambda col: (col-col.min())/(col.max()-col.min()))
            p = fake_models['random_forest'].predict(df2[fake_feats])[0]
            st.write('📌 Fake' if p==1 else '📌 Real')

elif option == '🤖 Bot Automated Detection':
    st.title('🤖 Automated Account Detection')
    upload2 = st.file_uploader('📁 Upload CSV for 🤖 Automated Detection', type=['csv'])
    if upload2:
        df = pd.read_csv(upload2)

        # If raw list-type columns exist, convert them to feature stats
        if any(col in df.columns for col in LIST_COLS):
            st.info("🔍 Detected raw list columns. Preprocessing them into features...")
            df = process_auto_raw(df)

        # Check for missing required features
        missing = [f for f in auto_feats if f not in df.columns]
        if missing:
            st.error(f'Missing features: {missing}')
        else:
            df[auto_feats] = auto_scaler.transform(df[auto_feats])
            preds = auto_models['RandomForest'].predict(df[auto_feats])
            df['Prediction'] = ['Automated' if p == 1 else 'Non-Automated' for p in preds]
            st.success("📁 Prediction completed successfully.")
            st.dataframe(df)
    else:
        st.subheader('🔍 Manual Input Not Available')
        st.write('📁 Batch CSV upload with either raw list columns or precomputed features is required.')



elif option == '📊 Fake Account Performance':
    st.title('📊 Fake Model Performance')
    for name, rep in fake_reports.items():
        st.markdown(f'### {name}')
        #st.dataframe(pd.DataFrame(rep).transpose())
        st.markdown('### Performance Comparison')
        df = pd.DataFrame(rep).transpose()
        st.dataframe(df)

        # Melt the DataFrame to long-form for seaborn
        df_long = df[['precision', 'recall', 'f1-score']].reset_index().melt(id_vars='index', 
                                                                             var_name='Metric', 
                                                                             value_name='Score')
        df_long.rename(columns={'index': 'Label'}, inplace=True)

        # Plotting
        st.markdown('### Performance Comparison')
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df_long, x='Metric', y='Score', hue='Label', palette='Set2', ax=ax)
        ax.set_title(f'{name} Metrics by Class')
        ax.set_ylim(0, 1.05)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="edge")
        st.pyplot(fig)
        


elif option == '📊 Bot Automated Performance':
    st.title('📊 Automated Model Performance')
    for name, rep in auto_reports.items():
        st.markdown(f'### {name}')
        #st.dataframe(pd.DataFrame(rep).transpose())
        df = pd.DataFrame(rep).transpose()
        st.dataframe(df)

        # Melt the DataFrame to long-form for seaborn
        df_long = df[['precision', 'recall', 'f1-score']].reset_index().melt(id_vars='index', 
                                                                             var_name='Metric', 
                                                                             value_name='Score')
        df_long.rename(columns={'index': 'Label'}, inplace=True)

        # Plotting
        st.markdown('### Performance Comparison')
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df_long, x='Metric', y='Score', hue='Label', palette='Set2', ax=ax)
        ax.set_title(f'{name} Metrics by Class')
        ax.set_ylim(0, 1.05)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="edge")
        st.pyplot(fig)
        
