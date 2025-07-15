import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import (
    VarianceThreshold, 
    SelectKBest, 
    f_regression, 
    mutual_info_regression,
    RFE,
    SequentialFeatureSelector
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('data-set.csv')
print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Target variable (SalePrice) stats:")
print(df['SalePrice'].describe())

def feature_engineering_and_encoding(df):
    """
    Complete feature engineering and encoding following class methodology
    """
    df_processed = df.copy()
    
    print("=== Feature Engineering ===")
    
    # 1. FEATURE ENGINEERING (from class examples)
    
    # Numeric Combinations (from class: ratios, differences, etc.)
    df_processed['PricePerSqFt'] = df_processed['SalePrice'] / df_processed['GrLivArea']
    df_processed['HouseAge'] = df_processed['YrSold'] - df_processed['YearBuilt']
    df_processed['YearsSinceRemod'] = df_processed['YrSold'] - df_processed['YearRemodAdd']
    df_processed['TotalSF'] = df_processed['TotalBsmtSF'] + df_processed['1stFlrSF'] + df_processed['2ndFlrSF']
    df_processed['TotalBathrooms'] = df_processed['FullBath'] + df_processed['HalfBath'] + df_processed['BsmtFullBath'] + df_processed['BsmtHalfBath']
    df_processed['TotalPorchSF'] = df_processed['OpenPorchSF'] + df_processed['EnclosedPorch'] + df_processed['3SsnPorch'] + df_processed['ScreenPorch']
    
    # Interaction Features (from class: Age × BMI example)
    df_processed['QualityAge'] = df_processed['OverallQual'] * df_processed['HouseAge']
    df_processed['QualityArea'] = df_processed['OverallQual'] * df_processed['GrLivArea']
    
    # Temporal Features (from class: Year, Month, Day of week, Season)
    df_processed['SaleMonth'] = df_processed['MoSold']
    df_processed['SaleYear'] = df_processed['YrSold']
    df_processed['IsWinterSale'] = (df_processed['MoSold'].isin([12, 1, 2])).astype(int)
    df_processed['IsSpringSale'] = (df_processed['MoSold'].isin([3, 4, 5])).astype(int)
    df_processed['IsSummerSale'] = (df_processed['MoSold'].isin([6, 7, 8])).astype(int)
    df_processed['IsFallSale'] = (df_processed['MoSold'].isin([9, 10, 11])).astype(int)
    
    print(f"Added {len(['PricePerSqFt', 'HouseAge', 'YearsSinceRemod', 'TotalSF', 'TotalBathrooms', 'TotalPorchSF', 'QualityAge', 'QualityArea', 'IsWinterSale', 'IsSpringSale', 'IsSummerSale', 'IsFallSale'])} engineered features")
    
    print("\n=== Categorical Encoding ===")
    
    # 2. ORDINAL ENCODING (from class list)
    ordinal_mappings = {
        'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'PoolQC': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4},
        'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3},
        'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
        'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
        'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
        'Fence': {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
    }
    
    ordinal_encoded_count = 0
    for feature, mapping in ordinal_mappings.items():
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna('NA')
            df_processed[f'{feature}_Encoded'] = df_processed[feature].map(mapping).fillna(0)
            ordinal_encoded_count += 1
    
    print(f"Applied ordinal encoding to {ordinal_encoded_count} features")
    
    # 3. BINARY ENCODING (from class: for features with 2 values)
    binary_features = ['CentralAir', 'Street']
    binary_encoded_count = 0
    for feature in binary_features:
        if feature in df_processed.columns:
            df_processed[f'{feature}_Binary'] = (df_processed[feature] == 'Y').astype(int) if feature == 'CentralAir' else (df_processed[feature] == 'Pave').astype(int)
            binary_encoded_count += 1
    
    print(f"Applied binary encoding to {binary_encoded_count} features")
    
    # 4. TARGET ENCODING (from class: mean value per category)
    target_encode_features = ['Neighborhood', 'MSSubClass', 'MSZoning', 'SaleType', 'SaleCondition']
    target_encoded_count = 0
    for feature in target_encode_features:
        if feature in df_processed.columns:
            target_mean = df_processed.groupby(feature)['SalePrice'].mean()
            df_processed[f'{feature}_TargetEncoded'] = df_processed[feature].map(target_mean)
            target_encoded_count += 1
    
    print(f"Applied target encoding to {target_encoded_count} features")
    
    # 5. ONE-HOT ENCODING using pd.get_dummies (from class)
    # Select categorical features with reasonable cardinality for one-hot encoding
    categorical_cols = ['BldgType', 'HouseStyle', 'RoofStyle', 'Foundation', 'Heating', 'Electrical']
    one_hot_features = []
    
    for col in categorical_cols:
        if col in df_processed.columns and df_processed[col].nunique() <= 8:
            one_hot_features.append(col)
    
    if one_hot_features:
        # Use pd.get_dummies as shown in class
        one_hot_encoded = pd.get_dummies(df_processed[one_hot_features], prefix=one_hot_features, drop_first=True)
        df_processed = pd.concat([df_processed, one_hot_encoded], axis=1)
        print(f"Applied one-hot encoding to {len(one_hot_features)} features, created {one_hot_encoded.shape[1]} dummy variables")
    
    # Remove original categorical columns and keep only numerical features
    numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    df_final = df_processed[numerical_features].copy()
    
    print(f"\nFinal dataset shape: {df_final.shape}")
    print(f"Total features created: {df_final.shape[1]}")
    
    return df_final

def step1_variance_threshold(X, threshold=0.0):
    """
    Step 1: Purge constants with variance threshold (class workflow)
    """
    print("\n=== Step 1: Variance Threshold Filter ===")
    
    # Remove features with zero or very low variance
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()]
    removed_features = X.columns[~selector.get_support()]
    
    print(f"Original features: {X.shape[1]}")
    print(f"Features after variance filtering: {len(selected_features)}")
    print(f"Removed {len(removed_features)} constant/low-variance features")
    
    if len(removed_features) > 0:
        print("Removed features:", list(removed_features)[:5], "..." if len(removed_features) > 5 else "")
    
    return pd.DataFrame(X_filtered, columns=selected_features, index=X.index)

def step2_filter_methods(X, y):
    """
    Step 2: Filter Methods - Pearson Correlation and Mutual Information
    """
    print("\n=== Step 2: Filter Methods ===")
    
    results = {}
    
    # METHOD 1: Pearson Correlation (from class formula)
    print("\nMethod 1: Pearson Correlation")
    print("Formula: r = cov(x, y) / (σₓ σᵧ)")
    
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    
    # Filter features with correlation > 0.3 (reasonable threshold)
    correlation_threshold = 0.3
    pearson_features = correlations[correlations > correlation_threshold].index.tolist()
    
    print(f"Features with |correlation| > {correlation_threshold}: {len(pearson_features)}")
    print("Top 15 correlated features:")
    for i, (feature, corr) in enumerate(correlations.head(15).items(), 1):
        print(f"{i:2d}. {feature:<25}: {corr:.4f}")
    
    results['pearson'] = {
        'features': pearson_features,
        'scores': correlations[pearson_features],
        'all_correlations': correlations
    }
    
    # METHOD 2: Mutual Information (from class formula)
    print(f"\nMethod 2: Mutual Information")
    print("Formula: MI = Σ p(x,y) log [p(x,y)/(p(x)p(y))]")
    
    # Use top correlated features to reduce computational cost
    top_features_for_mi = correlations.head(30).index.tolist()
    # Fill NaNs with median for MI selection
    X_mi = X[top_features_for_mi].fillna(X[top_features_for_mi].median())
    mi_selector = SelectKBest(score_func=mutual_info_regression, k=15)
    mi_selector.fit(X_mi, y)
    
    mi_features = [top_features_for_mi[i] for i in range(len(top_features_for_mi)) if mi_selector.get_support()[i]]
    mi_scores = mi_selector.scores_[mi_selector.get_support()]
    
    print(f"Top 15 features by Mutual Information:")
    for i, (feature, score) in enumerate(zip(mi_features, mi_scores), 1):
        print(f"{i:2d}. {feature:<25}: {score:.4f}")
    
    results['mutual_info'] = {
        'features': mi_features,
        'scores': dict(zip(mi_features, mi_scores))
    }
    
    return results

def step3_wrapper_methods(X, y, selected_features):
    """
    Step 3: Wrapper Methods - RFE and Sequential Selection
    """
    print("\n=== Step 3: Wrapper Methods ===")
    
    # Use selected features from filter methods
    X_selected = X[selected_features]
    # Fill NaNs with median for wrapper methods
    X_selected = X_selected.fillna(X_selected.median())
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # METHOD 1: Recursive Feature Elimination (from class table)
    print("\nMethod 1: Recursive Feature Elimination (RFE)")
    print("Iteratively fits model, drops least important feature, refits until k remain")
    
    # Use RandomForest for feature importance (tree-based from class)
    rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe_selector = RFE(estimator=rf_estimator, n_features_to_select=12, step=1)
    rfe_selector.fit(X_train, y_train)
    
    rfe_features = X_selected.columns[rfe_selector.get_support()].tolist()
    
    # Evaluate performance
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train[rfe_features], y_train)
    rfe_pred = rf_model.predict(X_test[rfe_features])
    rfe_r2 = r2_score(y_test, rfe_pred)
    
    print(f"RFE selected {len(rfe_features)} features")
    print(f"R² Score: {rfe_r2:.4f}")
    print("Selected features:")
    for i, feature in enumerate(rfe_features, 1):
        print(f"{i:2d}. {feature}")
    
    results['rfe'] = {
        'features': rfe_features,
        'r2_score': rfe_r2,
        'method': 'Recursive Feature Elimination'
    }
    
    # METHOD 2: Sequential Forward Selection (from class table)
    print(f"\nMethod 2: Sequential Forward Selection (SFS)")
    print("Greedy build-up: start empty, add feature that most improves score")
    
    sfs_estimator = LinearRegression()
    sfs_selector = SequentialFeatureSelector(
        estimator=sfs_estimator, 
        n_features_to_select=12, 
        direction='forward',
        cv=3,
        scoring='r2'
    )
    sfs_selector.fit(X_train, y_train)
    
    sfs_features = X_selected.columns[sfs_selector.get_support()].tolist()
    
    # Evaluate performance
    lr_model = LinearRegression()
    lr_model.fit(X_train[sfs_features], y_train)
    sfs_pred = lr_model.predict(X_test[sfs_features])
    sfs_r2 = r2_score(y_test, sfs_pred)
    
    print(f"SFS selected {len(sfs_features)} features")
    print(f"R² Score: {sfs_r2:.4f}")
    print("Selected features:")
    for i, feature in enumerate(sfs_features, 1):
        print(f"{i:2d}. {feature}")
    
    results['sfs'] = {
        'features': sfs_features,
        'r2_score': sfs_r2,
        'method': 'Sequential Forward Selection'
    }
    
    return results

def analyze_overlap_and_visualize(filter_results, wrapper_results):
    """
    Analyze feature overlap and create visualizations
    """
    print("\n=== Feature Overlap Analysis ===")
    
    # Collect all methods
    all_methods = {
        'Pearson': set(filter_results['pearson']['features']),
        'Mutual_Info': set(filter_results['mutual_info']['features']),
        'RFE': set(wrapper_results['rfe']['features']),
        'SFS': set(wrapper_results['sfs']['features'])
    }
    
    # Calculate overlap matrix
    methods = list(all_methods.keys())
    overlap_matrix = pd.DataFrame(index=methods, columns=methods, dtype=int)
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                overlap_matrix.loc[method1, method2] = len(all_methods[method1])
            else:
                overlap = len(all_methods[method1].intersection(all_methods[method2]))
                overlap_matrix.loc[method1, method2] = overlap
    
    print("Overlap Matrix (number of common features):")
    print(overlap_matrix)
    
    # Find common features across all methods
    common_features = set.intersection(*all_methods.values())
    print(f"\nFeatures selected by ALL methods ({len(common_features)}):")
    for feature in sorted(common_features):
        print(f"- {feature}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Correlation heatmap
    plt.subplot(2, 2, 1)
    top_features = filter_results['pearson']['all_correlations'].head(15)
    plt.barh(range(len(top_features)), top_features.values)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Correlation with SalePrice')
    plt.title('Top 15 Features - Pearson Correlation')
    plt.gca().invert_yaxis()
    
    # Plot 2: Method overlap heatmap
    plt.subplot(2, 2, 2)
    sns.heatmap(overlap_matrix.astype(float), annot=True, fmt='.0f', cmap='Blues', cbar=True)
    plt.title('Feature Selection Method Overlap')
    plt.xlabel('Methods')
    plt.ylabel('Methods')
    
    # Plot 3: R² scores comparison
    plt.subplot(2, 2, 3)
    r2_scores = {
        'RFE': wrapper_results['rfe']['r2_score'],
        'SFS': wrapper_results['sfs']['r2_score']
    }
    plt.bar(r2_scores.keys(), r2_scores.values(), color=['skyblue', 'lightcoral'])
    plt.ylabel('R² Score')
    plt.title('Model Performance by Wrapper Method')
    plt.ylim(0, 1)
    for i, (method, score) in enumerate(r2_scores.items()):
        plt.text(i, score + 0.01, f'{score:.4f}', ha='center')
    
    # Plot 4: Feature count by method
    plt.subplot(2, 2, 4)
    feature_counts = {method: len(features) for method, features in all_methods.items()}
    plt.bar(feature_counts.keys(), feature_counts.values(), color=['lightgreen', 'orange', 'skyblue', 'lightcoral'])
    plt.ylabel('Number of Selected Features')
    plt.title('Feature Count by Selection Method')
    plt.xticks(rotation=45)
    for i, (method, count) in enumerate(feature_counts.items()):
        plt.text(i, count + 0.3, str(count), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return overlap_matrix, common_features

# MAIN EXECUTION
if __name__ == "__main__":
    try:
        print("=== HOUSE PRICING FEATURE SELECTION ANALYSIS ===")
        print("Following Class Methodology: Filter → Wrapper Methods")
        # Step 0: Feature Engineering and Encoding
        df_processed = feature_engineering_and_encoding(df)
        # Separate features and target
        X = df_processed.drop(['SalePrice', 'Id'], axis=1, errors='ignore')
        y = df_processed['SalePrice']
        print(f"\nFinal feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        # Step 1: Variance Threshold
        X_filtered = step1_variance_threshold(X, threshold=0.0)
        # Step 2: Filter Methods
        filter_results = step2_filter_methods(X_filtered, y)
        # Combine features from both filter methods for wrapper methods
        combined_features = list(set(filter_results['pearson']['features'] + filter_results['mutual_info']['features']))
        print(f"\nCombined features from filter methods: {len(combined_features)}")
        # Step 3: Wrapper Methods
        wrapper_results = step3_wrapper_methods(X_filtered, y, combined_features)
        # Step 4: Analysis and Visualization
        overlap_matrix, common_features = analyze_overlap_and_visualize(filter_results, wrapper_results)
        print("\n=== SUMMARY ===")
        print(f"Total original features: {df.shape[1]-1}")  # -1 for target
        print(f"Total engineered features: {X.shape[1]}")
        print(f"Features after variance filtering: {X_filtered.shape[1]}")
        print(f"Features selected by Pearson: {len(filter_results['pearson']['features'])}")
        print(f"Features selected by Mutual Info: {len(filter_results['mutual_info']['features'])}")
        print(f"Features selected by RFE: {len(wrapper_results['rfe']['features'])}")
        print(f"Features selected by SFS: {len(wrapper_results['sfs']['features'])}")
        print(f"Common features across all methods: {len(common_features)}")
        print(f"\nBest R² Score: {max(wrapper_results['rfe']['r2_score'], wrapper_results['sfs']['r2_score']):.4f}")
        # Print final feature recommendations
        print(f"\n=== RECOMMENDED FINAL FEATURES ===")
        if len(common_features) > 0:
            print("Features consistently selected across all methods:")
            for feature in sorted(common_features):
                print(f"- {feature}")
        else:
            print("No features selected by all methods. Consider features from best performing method:")
            best_method = 'rfe' if wrapper_results['rfe']['r2_score'] > wrapper_results['sfs']['r2_score'] else 'sfs'
            for feature in wrapper_results[best_method]['features']:
                print(f"- {feature}")

        # === Visualize most important features ===
        try:
            core = sorted(common_features)
            extras = [f for f in wrapper_results['rfe']['features'] if f not in core]
            labels = core + extras
            y_vals = [1]*len(labels)
            colors = ['tab:green']*len(core) + ['tab:orange']*len(extras)
            plt.figure(figsize=(10,4))
            plt.bar(labels, y_vals, color=colors)
            plt.xticks(rotation=45, ha='right')
            plt.yticks([])
            plt.title('Most Important Features Selected')
            for i, lbl in enumerate(core):
                plt.text(i, 1.02, 'core', ha='center', fontsize=8)
            plt.tight_layout()
            plt.savefig('important_features.png')
            print("\nSaved visual: important_features.png")
        except Exception as viz_e:
            print('Visualization failed:', viz_e)

    except Exception as e:
        import traceback
        print("An error occurred:")
        traceback.print_exc()