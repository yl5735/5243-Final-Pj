from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib_venn import venn2, venn3
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import numpy as np

# Load raw and cleaned feature matrix and target labels
df_raw = pd.read_csv("dataset/df_raw.csv")
df_features = pd.read_csv("dataset/features.csv")
df_target = pd.read_csv("dataset/target.csv")

# Feature selection outputs
feature_selections = {
    "Pearson": [
        "PctKids2Par", "PctFam2Par", "racePctWhite", "PctYoungKids2Par", "PctTeen2Par",
        "pctWInvInc", "FemalePctDiv", "pctWPubAsst", "PctIlleg", "NumIlleg"
    ],
    "Mutual Information": [
        "PctKids2Par", "PctIlleg", "PctFam2Par", "racePctWhite", "NumIlleg",
        "PctTeen2Par", "PctYoungKids2Par", "NumUnderPov", "PctPopUnderPov", "pctWInvInc"
    ],
    "Stepwise Forward": [
        "racepctblack", "racePctWhite", "racePctHisp", "agePct12t29", "agePct16t24",
        "numbUrban", "pctUrban", "pctWWage", "pctWFarmSelf", "pctWInvInc"
    ],
    "Stepwise Backward": [
        "racepctblack", "racePctHisp", "agePct12t29", "pctUrban", "pctWWage",
        "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWRetire", "medFamInc"
    ],
    "Lasso": [
        "racepctblack", "racePctWhite", "pctUrban", "pctWInvInc", "OtherPerCap",
        "MalePctDivorce", "PctKids2Par", "PctWorkMom", "PctIlleg", "PctPersDenseHous"
    ],
    "ElasticNet": [
        "racepctblack", "racePctWhite", "pctUrban", "pctWWage", "pctWInvInc",
        "OtherPerCap", "MalePctDivorce", "PctKids2Par", "PctWorkMom", "PctIlleg"
    ],
    "OLS": [
        "racepctblack", "pctUrban", "pctWFarmSelf", "pctWInvInc", "pctWRetire",
        "whitePerCap", "PctPopUnderPov", "PctEmploy", "PctEmplManu", "MalePctDivorce"
    ]
}

regression_results = {
    'Lasso': {'MSE': 0.01712, 'RMSE': 0.13084, 'MAE': 0.09461, 'R2': 0.69694, 'Adjusted R2': 0.69387},
    'Ridge': {'MSE': 0.01705, 'RMSE': 0.13057, 'MAE': 0.09450, 'R2': 0.69819, 'Adjusted R2': 0.69513},
    'Elastic Net': {'MSE': 0.01709, 'RMSE': 0.13074, 'MAE': 0.09456, 'R2': 0.69739, 'Adjusted R2': 0.69433},
    'Random Forest': {'MSE': 0.01652, 'RMSE': 0.12854, 'MAE': 0.09260, 'R2': 0.70751, 'Adjusted R2': 0.69526},
    'SVR': {'MSE': 0.02155, 'RMSE': 0.14680, 'MAE': 0.10587, 'R2': 0.61847, 'Adjusted R2': 0.60249},
    'XGBRegressor': {'MSE': 0.01718, 'RMSE': 0.13109, 'MAE': 0.09174, 'R2': 0.69577, 'Adjusted R2': 0.68303}
}

error_df = pd.DataFrame(regression_results).T.reset_index().rename(columns={'index': 'Model'})
classification_results = {
    "Logistic Regression": {
        "Accuracy": 0.88221,
        "ROC AUC": 0.93305,
        "Precision": [0.90, 0.83],
        "Recall": [0.95, 0.71],
        "F1": [0.92, 0.77],
        "Support": [291, 108],
        "Confusion": [[275, 16], [31, 77]]
    },
    "Random Forest": {
        "Accuracy": 0.88221,
        "ROC AUC": 0.92669,
        "Precision": [0.91, 0.79],
        "Recall": [0.92, 0.77],
        "F1": [0.92, 0.78],
        "Support": [291, 108],
        "Confusion": [[269, 22], [25, 83]]
    },
    "XGBoost": {
        "Accuracy": 0.88,
        "ROC AUC": 0.9240,
        "Precision": [0.91, 0.81],
        "Recall": [0.93, 0.74],
        "F1": [0.92, 0.77],
        "Support": [291, 108],
        "Confusion": [[272, 19], [28, 80]]
    },
    "CatBoost": {
        "Accuracy": 0.88,
        "ROC AUC": 0.9256,
        "Precision": [0.91, 0.81],
        "Recall": [0.93, 0.75],
        "F1": [0.92, 0.78],
        "Support": [291, 108],
        "Confusion": [[272, 19], [27, 81]]
    }
}


def plot_confusion_matrix(confusion, model_name):
    df = pd.DataFrame(confusion, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])
    fig, ax = plt.subplots()
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues" if "XGBoost" in model_name else "Greens", ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix (Threshold = 0.45)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    return fig

app_ui = ui.page_fluid(
    ui.navset_tab(

        # Tab 1: Intro
        ui.nav_panel("Intro", ui.markdown("""
        ### Welcome to US Crime Rate Modeling & Prediction Dashboard  
        This interactive dashboard is designed to explore, model, and predict violent crime rates based on socioeconomic and community-level features. It guides users through the full data science pipeline — from exploratory data analysis to feature selection, and from regression to classification modeling. Use the navigation tabs to explore cleaned datasets, visualize patterns, evaluate model performance, and compare prediction results across multiple algorithms.
        """)),

        # Tab 2: Data Output
        ui.nav_panel("Data Output",
            ui.markdown("""
            This section displays the raw dataset. You can optionally dealing with missing values, apply label encoding to categorical features, or standardize numeric features.

            In preparation for unsupervised learning in the EDA phase and further machine learning tasks:
            - Identification columns like State, County, Community, and their names are removed
            - Variables are separated by datatype (numerical or categorical)
            - Numerical features are standardized
            """),

            ui.input_checkbox("drop_na", "Drop rows with missing values", False),
            ui.input_checkbox("label_encode", "Apply label encoding to categorical features", False),
            ui.input_checkbox("standardize", "Standardize numeric features", False),

            ui.output_table("cleaned_table")
        ),

        # Tab 3: EDA
        ui.nav_panel("EDA",
            ui.navset_tab(
                ui.nav_panel("Target",
                    ui.markdown("This section explores the target variable `ViolentCrimesPerPop` through summary statistics and visualizations, including its geographic distribution across the U.S."),
                    ui.output_table("y_summary"),
                    ui.row(
                        ui.column(6,
                            ui.output_image("img_distribution"),
                            ui.markdown("""
                            From the distribution histogram and KDE curve, the data is highly right-skewed. Most communities have low violent crime rates (peaking around 0.1), with a long tail toward high crime values. This suggests a non-normal distribution and potential need for transformation in modeling.
                            """)
                        ),
                        ui.column(6,
                            ui.output_image("img_boxplot"),
                            ui.markdown("""
                            The boxplot shows a median near 0.2, and many data points above 0.6 are flagged as outliers. Some extreme values reach 1.0, indicating a small group of communities with extremely high violence rates.
                            """)
                        )
                    ),
                    ui.row(
                        ui.column(6,
                            ui.output_image("img_map_highcrime"),
                            ui.markdown("""
                            This map displays outlier communities with high violent crime rates. These are mainly clustered in the southeastern U.S., especially in Florida, Georgia, Alabama, and South Carolina.
                            """)
                        ),
                        ui.column(6,
                            ui.output_image("img_state"),
                            ui.markdown("""
                            The choropleth map shows the average violent crime rate by state. Darker shades represent higher state-level crime rates, largely overlapping with regions where outlier communities are found.
                            """)
                        )
                    )
                ),
                ui.nav_panel("Cluster",
                    ui.markdown("""
                    Clustering was used to perform further exploratory data analysis. The Elbow Method and Silhouette Analysis were applied to determine the optimal number of clusters, both of which indicated that three clusters would yield the best results. Based on this, t-SNE was used for dimensionality reduction and cluster visualization.
                    
                    The visualization results show three clearly defined and well-separated clusters. Among them, the blue cluster (cluster 2) is distinctly different from the other two, possibly representing a uniquely structured group. The orange and green clusters (clusters 0 and 1) exhibit some boundary overlap but are generally well separated. The t-SNE visualization further confirms the validity of the k=3 clustering structure, indicating that the data indeed contain three groups with significant structural differences. Given the stability and clarity of the clustering results, this approach can be useful for identifying distinct social groupings. As such, the cluster labels can be considered as a new categorical feature in subsequent modeling, enabling a semi-supervised approach to classification.
                    """),
                    ui.output_image("img_elbow_silhouette"),
                    ui.output_image("img_tsne")
                ),
            )
        ),

        # Tab 4: Feature Selection
        ui.nav_panel("Feature Selection",ui.markdown("""
            In this section, you can compare top features by selecting different methods.  
            Select 1 to 3 methods to view their top 10 features, and overlap in a Venn diagram.  
            A summary table also shows which features are shared or unique.
            """),
            ui.input_checkbox_group("fs_methods", "Choose one or more methods:",
                choices=list(feature_selections.keys()), selected=["Pearson"]),
            ui.output_ui("fs_top10_text"),
            ui.output_table("fs_feature_summary"),
            ui.output_plot("fs_venn_plot")

        ),

        # Tab 5: Regression Modeling
        ui.nav_panel("Regression Modeling",
            ui.markdown("""
            This section compares regression model performance across several methods.  
            Select one or more models to view metrics and choose which error type to visualize.  
            The image below compares different feature selection methods. For this graph, we ran all methods 10 times and selected the most stable method with the smallest MSE.
            """),
            ui.output_image("img_model_comparison"),

            # Now continue with controls in a clean vertical stack
            ui.input_checkbox_group("selected_models", "Select regression models:",
                choices=list(regression_results.keys()), selected=["lasso"]),
            ui.output_table("reg_metrics"),
            ui.input_select("error_type", "Choose error to plot:",
                choices=["MSE", "RMSE", "MAE", "R2", "Adjusted R2"], selected="MSE"),
            ui.output_plot("error_plot")
        ),

        # Tab 6: Classification Modeling
            ui.nav_panel("Classification Modeling",
                ui.markdown("""
                This section displays classification model results using fixed threshold = 0.45.  
                Metrics such as accuracy, precision, recall, and F1 are shown below for each model.
                """),
                
                # Model selection dropdown
                ui.input_select("model", "Choose a model:", list(classification_results.keys())),
                
                # Side-by-side layout
                ui.layout_columns(
                    ui.card(
                        ui.h5("Classification Metrics"),
                        ui.output_table("clf_table")
                    ),
                    ui.card(
                        ui.h5("Confusion Matrix"),
                        ui.output_plot("clf_matrix", width="80%")
                    )
                )
            )
    )
)

def server(input, output, session):

    @output
    @render.table
    def cleaned_table():
        df = df_raw.copy()

        if input.drop_na():
            df = df.dropna()

        if input.label_encode():
            label_encoder = LabelEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = label_encoder.fit_transform(df[col].astype(str))

        if input.standardize():
            # Drop identifier columns
            id_columns = ["state", "county", "community", "communityname", "fold"]
            df = df.drop(columns=[col for col in id_columns if col in df.columns], errors='ignore')
            numeric_cols = df.select_dtypes(include=['number']).columns
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df.head(50)

    @output
    @render.table
    def y_summary():
        stats = {"count": [1994], "mean": [0.237979], "std": [0.232985], "min": [0.000000], "25%": [0.070000], "50%": [0.150000],"75%": [0.330000], "max": [1.000000]}
        return pd.DataFrame(stats).round(6)

    @output
    @render.image
    def img_distribution():
        return {"src": "image/distribution.png", "width": "500px"}
    
    @output
    @render.image
    def img_boxplot():
        return {"src": "image/boxplot.png", "width": "500px"}

    @output
    @render.image
    def img_map_highcrime():
        return {"src": "image/map_highcrime.png", "width": "500px"}

    @output
    @render.image
    def img_state():
        return {"src": "image/map_state.png", "width": "500px"}

    @output
    @render.image
    def img_model_comparison():
        return  {"src": "image/model_comparison.png", "width": "800px"}
    
    @output
    @render.image
    def img_elbow_silhouette():
        return  {"src": "image/elbow_silhouette.png", "width": "800px"}

    @output
    @render.image
    def img_tsne():
        return  {"src": "image/tsne.png", "width": "500px"}

    @output
    @render.ui
    def fs_top10_text():
        methods = input.fs_methods()
        if not methods:
            return ui.markdown("**Please select at least one method.**")

        cols = []
        for m in methods:
            feats = feature_selections[m]
            box = ui.div(
                ui.markdown(f"#### {m}"),
                ui.tags.ul([ui.tags.li(f) for f in feats]),
                class_="p-2 border"
            )
            cols.append(box)

        return ui.layout_columns(*cols)

    @output
    @render.plot
    def fs_venn_plot():
        methods = input.fs_methods()
        fig, ax = plt.subplots()

        if len(methods) == 2:
            set1 = set(feature_selections[methods[0]])
            set2 = set(feature_selections[methods[1]])
            v = venn2([set1, set2], set_labels=methods, ax=ax)
            ax.set_title("Overlap of Top 10 Important Features (Feature Names Inside)")
            for subset in ('10', '01', '11'):
                label = v.get_label_by_id(subset)
                if label:
                    ids = set1 if subset == '10' else set2 if subset == '01' else set1 & set2
                    label.set_text("\n".join(sorted(ids)))

        elif len(methods) == 3:
            sets = [set(feature_selections[m]) for m in methods]
            v = venn3(sets, set_labels=methods, ax=ax)
            ax.set_title("Overlap of Top 10 Important Features (Feature Names Inside)")
            region_ids = ['100', '010', '001', '110', '101', '011', '111']
            logic_map = {
                '100': lambda s: sets[0] - sets[1] - sets[2],
                '010': lambda s: sets[1] - sets[0] - sets[2],
                '001': lambda s: sets[2] - sets[0] - sets[1],
                '110': lambda s: sets[0] & sets[1] - sets[2],
                '101': lambda s: sets[0] & sets[2] - sets[1],
                '011': lambda s: sets[1] & sets[2] - sets[0],
                '111': lambda s: sets[0] & sets[1] & sets[2],
            }
            for region in region_ids:
                label = v.get_label_by_id(region)
                if label:
                    items = logic_map[region](sets)
                    label.set_text("\n".join(sorted(items)))
        else:
            ax.text(0.5, 0.5, "Select 2 or 3 methods to show Venn diagram.", ha='center', va='center')
            ax.axis('off')

        return fig

    @output
    @render.table
    def fs_feature_summary():
        methods = input.fs_methods()
        if not methods:
            return pd.DataFrame({"Group": [], "Features": []})

        all_features = []
        for m in methods:
            all_features.extend(feature_selections[m])
        counts = Counter(all_features)

        groupings = {"In All Methods": [], "Shared by 2": [], "Unique": []}
        for feat, count in counts.items():
            if count == len(methods):
                groupings["In All Methods"].append(feat)
            elif count == 2:
                groupings["Shared by 2"].append(feat)
            elif count == 1:
                groupings["Unique"].append(feat)

        summary = []
        for group, feats in groupings.items():
            summary.append({"Group": group, "Features": ", ".join(sorted(feats))})

        return pd.DataFrame(summary)


# Regression
    @output
    @render.table
    def reg_metrics():
        selected = input.selected_models()
        if not selected:
            return pd.DataFrame()
        df = error_df[error_df['Model'].isin(selected)].copy()
        return df.round(5)

    @output
    @render.plot
    def error_plot():
        selected = input.selected_models()
        if not selected:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Please select at least one model", ha='center')
            ax.axis('off')
            return fig

        metric = input.error_type()
        fig, ax = plt.subplots()
        sns.barplot(data=error_df[error_df['Model'].isin(selected)], x="Model", y=metric, hue="Model", ax=ax, legend=False, palette="Set2")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} plot")
        return fig

    # Classification
    @output
    @render.table
    def clf_table():
        model = input.model()
        result = classification_results[model]

        df = pd.DataFrame({
            "precision": result["Precision"],
            "recall": result["Recall"],
            "f1-score": result["F1"],
            "support": result["Support"]
        }, index=["0", "1"])

        support = np.array(result["Support"])
        precision = np.array(result["Precision"])
        recall = np.array(result["Recall"])
        f1 = np.array(result["F1"])
        total = support.sum()
        accuracy = np.trace(result["Confusion"]) / total

        df.loc["accuracy"] = ["", "", accuracy, total]
        df.loc["macro avg"] = [precision.mean(), recall.mean(), f1.mean(), total]
        df.loc["weighted avg"] = [
            np.average(precision, weights=support),
            np.average(recall, weights=support),
            np.average(f1, weights=support),
            total
        ]

        df = df.reset_index()
        df.rename(columns={"index": "class"}, inplace=True)

        # Coerce numeric columns to float
        numeric_cols = ["precision", "recall", "f1-score", "support"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return (
            df.style
            .format({col: "{:.3g}" for col in numeric_cols})
            .set_properties(**{"text-align": "right"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "right")]}])
        )

    @output
    @render.plot
    def clf_matrix():
        model_name = input.model()
        cm = np.array(classification_results[model_name]["Confusion"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name} (Threshold = 0.45)")
        plt.tight_layout()
        return plt.gcf()


app = App(app_ui, server)
