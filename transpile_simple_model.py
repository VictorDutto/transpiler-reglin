import pandas as pd
import joblib
import os

from sklearn.linear_model import LinearRegression

def create_model():
    df = pd.read_csv("Cancerpays.csv", sep=";")
    X = df[["Colon", "Stomach"]]
    y = df["Lung"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, "model.joblib")

def produce_linear_regression_c_code(ground_truth):

    model = joblib.load('model.joblib')
    len_tht = len(model.coef_) + 1

    tht = f"{model.intercept_}f,"
    for coef in model.coef_:
        tht += str(coef) + "f,"
    tht = tht.strip(",")

    prediction_code = f"float tht[{len_tht}] = {{{tht}}};"


    c_pred = ""
    for value in ground_truth:
        c_pred += str(value) + "f,"

    to_predict_c = c_pred.strip(",")

    code = f"""
    #include <stdio.h>

    {prediction_code}
    float prediction(float *features, int n_feature)
    {{
        float res = tht[0];

        for (int i = 0; i < n_feature; ++i)
            res += features[i] * tht[i+1];

        return res;
    }}
    int main()
    {{
        float to_predict[2] = {{{c_pred}}};

        printf("%f\\n", prediction(to_predict, 2));

        return 0;
    }}
    """

    with open("fichier.c", "w") as f:
        f.write(code)

    if os.system("gcc fichier.c -O3 -o main"):
        print("Compile error")

gt = [-0.0041649365241367, 0.0017850734344602]
create_model()
produce_linear_regression_c_code(gt)

os.system("gcc fichier.c -O3 -o main")