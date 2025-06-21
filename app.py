# decision_tree_app.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import make_moons,make_blobs,make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


st.title("Visulization Tool")

model = st.selectbox("Select Model", ("--select--" ,"DecisionTreeClassification", "LogisticRegression" , "Support Vector Classification"))

if model == 'DecisionTreeClassification':
    # Create dataset
    X, y = make_moons(n_samples=400, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Show original scatter plot at the beginning (always)
    st.subheader("🔍 Original Data Distribution")
    plt.style.use('fivethirtyeight')
    fig_raw = plt.figure(figsize=(6, 4))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='purple', label="Class 0", s=20)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label="Class 1", s=20)
    plt.legend()
    plt.title("Original Scatter Plot")
    plt.tight_layout()
    st.pyplot(fig_raw)

    # --- Sidebar ---
    st.sidebar.title("Hyperparameters")
    criteria = st.sidebar.selectbox("Criterion", ('gini', 'entropy', 'log_loss'))
    splitter = st.sidebar.selectbox("Splitter", ('best', 'random'))
    depth = st.sidebar.number_input("Max Depth", min_value=1, value=3)

    # Allow "None" for max_leaf_nodes
    use_none_max_leaf = st.sidebar.checkbox("Use None for Max Leaf Nodes", value=False)
    if use_none_max_leaf:
        max_leaf_nodes = None
    else:
        max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes", min_value=2, max_value=100, step=1, value=10)

    min_split = st.sidebar.slider("Min Samples Split", 2, X.shape[0], 2)
    min_leaf = st.sidebar.slider("Min Samples Leaf", 1, X.shape[0], 1)
    min_impurity = st.sidebar.number_input("Min Impurity Decrease", 0.0, 2.0, step=0.01)

    # --- Page Navigation ---
    page = st.sidebar.radio("Select View", ["Decision Boundary", "Decision Tree"])

    # --- Session State for model ---
    if 'clf' not in st.session_state:
        st.session_state.clf = None
        st.session_state.acc = None

    if st.sidebar.button("🚀 Run Algorithm"):
        clf = DecisionTreeClassifier(
            criterion=criteria,
            splitter=splitter,
            max_depth=depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity,
            random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.session_state.clf = clf
        st.session_state.acc = acc
        st.success(f"Model trained! Accuracy: {acc:.2f}")

    # --- Plot Based on Page Selection ---
    if st.session_state.clf is not None:
        clf = st.session_state.clf

        if page == "Decision Boundary":
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                                np.linspace(y_min, y_max, 500))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            st.subheader("🌈 Decision Boundary")
            fig = plt.figure(figsize=(7, 5))
            plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors='k', s=20)
            plt.title("Decision Tree Decision Boundary")
            st.pyplot(fig)

        elif page == "Decision Tree":
            st.subheader("🧠 Trained Decision Tree Structure")
            fig_tree = plt.figure(figsize=(20, 10))
            plot_tree(clf, filled=True, feature_names=["X1", "X2"], class_names=["0", "1"])
            st.pyplot(fig_tree)

    else:
        st.info("ℹ️ Train the model using 'Run Algorithm' to see decision boundary or tree view.")
        
elif model == "LogisticRegression":

    # --- Solver to Penalty Map ---
    solver_penalty_map = {
        'lbfgs': ['l2', None],
        'liblinear': ['l1', 'l2'],
        'newton-cg': ['l2', None],
        'newton-cholesky': ['l2', None],
        'sag': ['l2', None],
        'saga': ['elasticnet', 'l1', 'l2', None]
    }

    def load_initial_graph(dataset, ax):
        if dataset == "Binary":
            X, y = make_blobs(n_features=2, centers=2, random_state=6)
        elif dataset == "Multiclass":
            X, y = make_blobs(n_features=2, centers=4, random_state=2)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
        return X, y

    def draw_meshgrid(X):
        a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
        b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
        XX, YY = np.meshgrid(a, b)
        input_array = np.array([XX.ravel(), YY.ravel()]).T
        return XX, YY, input_array

    # --- Sidebar UI ---
    st.sidebar.markdown("### Logistic Regression Classifier")

    dataset = st.sidebar.selectbox('Select Dataset', ('Binary', 'Multiclass'))
    solver = st.sidebar.selectbox('Solver', list(solver_penalty_map.keys()))

    valid_penalties = solver_penalty_map[solver]
    penalty = st.sidebar.selectbox('Penalty', ['None' if p is None else p for p in valid_penalties])
    penalty_value = None if penalty == 'None' else penalty

    if penalty_value == 'elasticnet':
        l1_ratio = st.sidebar.slider('l1 Ratio (for elasticnet)', 0.0, 1.0, 0.5)
    else:
        l1_ratio = None

    c_input = float(st.sidebar.number_input('C', value=1.0))
    max_iter = int(st.sidebar.number_input('Max Iterations', value=100))

    # Automatically choose multiclass for multiclass data, skip if binary
    if dataset == "Multiclass":
        multi_class = 'multinomial'
    else:
        multi_class = 'auto'

    # --- Plot Dataset ---
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    X, y = load_initial_graph(dataset, ax)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    orig = st.pyplot(fig)

    # --- Run Algorithm ---
    if st.sidebar.button("Run Algorithm"):
        orig.empty()

        try:
            kwargs = {
                'penalty': penalty_value,
                'C': c_input,
                'solver': solver,
                'max_iter': max_iter,
                # 'multi_class': multi_class
            }
            if penalty_value == 'elasticnet':
                kwargs['l1_ratio'] = l1_ratio

            clf = LogisticRegression(**kwargs)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            XX, YY, input_array = draw_meshgrid(X)
            labels = clf.predict(input_array)

            ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
            plt.xlabel("Col1")
            plt.ylabel("Col2")
            st.pyplot(fig)

            st.subheader(f"✅ Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")

elif model == 'Support Vector Classification':
# Sidebar options for SVC
    st.sidebar.title("SVC Hyperparameters")
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, step=0.01, value=1.0)
    dataset_type = st.sidebar.selectbox("Dataset", ["make_classification (3 classes)", "make_moons (2 classes)"])

    # Create dataset based on selection
    if dataset_type == "make_classification (3 classes)":
        X, y = make_classification(
            n_samples=500,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            n_classes=3,
            class_sep=1.0,
            random_state=42
        )
    else:
        dataset_type == "make_moons (2 classes)"
        X, y = make_moons(n_samples=500, noise=0.8, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Plot raw data
    st.subheader("📊 Original Data Distribution")
    fig_raw = plt.figure(figsize=(6, 4))
    plt.style.use('fivethirtyeight')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="deep", edgecolor="k")
    plt.title("Data Visualization")
    st.pyplot(fig_raw)

    # Train model
    if st.sidebar.button("🚀 Train Model"):
        svc = SVC(kernel=kernel, C=C, decision_function_shape='ovr')
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model Trained! Accuracy: {acc:.2f}")

        # Plot decision regions with support vectors
        def plot_decision_regions_with_support_vectors(X, y, model, title):
            plt.figure(figsize=(8, 6))
            ax = plt.gca()
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                                np.linspace(y_min, y_max, 500))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
            sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="deep", edgecolor="k", s=60)
            # plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            #             s=120, facecolors='none', edgecolors='black', linewidths=1.5,
            #             marker='o', label='Support Vectors')
            plt.title(title)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.legend(loc='upper right')
            st.pyplot(plt.gcf())

        plot_decision_regions_with_support_vectors(X, y, svc,
            f"SVC Decision Boundary (Kernel='{kernel}', C={C})")

    
