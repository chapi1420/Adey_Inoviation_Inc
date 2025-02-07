import pandas as pd

class FeatureSelector:
    def __init__(self, target_col):
        self.target = target_col
        self.selected_features = None
        
    def variance_threshold(self, df, threshold=0.01):
        """Remove low-variance features"""
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        return df.iloc[:, selector.fit(df).get_support()]
        
    def anova_selection(self, df, k=20):
        """ANOVA F-value for classification"""
        from sklearn.feature_selection import SelectKBest, f_classif
        X = df.drop(columns=[self.target])
        y = df[self.target]
        return SelectKBest(f_classif, k=k).fit(X, y)
        
    def mutual_info_selection(self, df, k=15):
        """Mutual information for continuous features"""
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        X = df.drop(columns=[self.target])
        y = df[self.target]
        return SelectKBest(mutual_info_classif, k=k).fit(X, y)

    def generate_report(self, df, method='anova'):
        """Generate feature importance report"""
        if method == 'anova':
            selector = self.anova_selection(df)
        elif method == 'mutual_info':
            selector = self.mutual_info_selection(df)
            
        scores = pd.DataFrame({
            'feature': df.drop(columns=[self.target]).columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        return scores
    
# Correlation Analysis
def plot_correlation_matrix(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return upper[upper > threshold].stack().index.tolist()
# Feature Importance (Tree-based)
def random_forest_importance(df, target):
    from sklearn.ensemble import RandomForestClassifier
    X = df.drop(columns=[target])
    y = df[target]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
