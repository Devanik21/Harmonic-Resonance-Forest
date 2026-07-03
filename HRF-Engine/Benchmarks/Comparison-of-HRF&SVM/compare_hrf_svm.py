import os
import os
import pandas as pd

benchmark_results = {
    "EEG Eye State": {
        "HRF Ultimate (GPU)": 0.9266,
        "SVM (RBF)":          0.8533,
        "Random Forest":      0.8950,
        "XGBoost":            0.8950,
    },
    "Phoneme": {
        "HRF Ultimate (GPU)": 0.9200,
        "SVM (RBF)":          0.8166,
        "Random Forest":      0.9100,
        "XGBoost":            0.9150,
    },
    "Wall-Following Robot": {
        "HRF Ultimate (GPU)": 0.9966,
        "SVM (RBF)":          0.8850,
        "Random Forest":      0.9950,
        "XGBoost":            0.9966,
    },
    "Electricity": {
        "HRF Ultimate (GPU)": 0.8533,
        "SVM (RBF)":          0.7800,
        "Random Forest":      0.8400,
        "XGBoost":            0.8316,
    },
    "Gas Sensor Drift": {
        "HRF Ultimate (GPU)": 0.9883,
        "SVM (RBF)":          0.9366,
        "Random Forest":      0.9883,
        "XGBoost":            0.9783,
    },
    "Japanese Vowels": {
        "HRF Ultimate (GPU)": 0.9800,
        "SVM (RBF)":          0.9783,
        "Random Forest":      0.9433,
        "XGBoost":            0.9500,
    },
    "Gesture Phase": {
        "HRF Ultimate (GPU)": 0.7033,
        "SVM (RBF)":          0.5500,
        "Random Forest":      0.6816,
        "XGBoost":            0.6716,
    },
    "Mfeat-Fourier": {
        "HRF Ultimate (GPU)": 0.8950,
        "SVM (RBF)":          0.8775,
        "Random Forest":      0.8575,
        "XGBoost":            0.8725,
    },
    "Optdigits": {
        "HRF Ultimate (GPU)": 0.9816,
        "SVM (RBF)":          0.9900,
        "Random Forest":      0.9916,
        "XGBoost":            0.9850,
    },
    "Solar Flare Evolution": {
        "HRF Ultimate (GPU)": 0.7777,
        "SVM (RBF)":          0.7777,
        "Random Forest":      0.7460,
        "XGBoost":            0.7301,
    },
    "Texture Analysis": {
        "HRF Ultimate (GPU)": 1.0000,
        "SVM (RBF)":          0.9046,
        "Random Forest":      0.9826,
        "XGBoost":            0.9971,
    },
    "Steel Plates Faults": {
        "HRF Ultimate (GPU)": 1.0000,
        "SVM (RBF)":          0.9948,
        "Random Forest":      0.9922,
        "XGBoost":            1.0000,
    },
    "HTRU2 Pulsar Detection": {
        "HRF Ultimate (GPU)": 0.7668,
        "SVM (RBF)":          0.7772,
        "Random Forest":      0.7668,
        "XGBoost":            0.7772,
    },
    "Madelon": {
        "HRF Ultimate (GPU)": 0.8500,
        "SVM (RBF)":          0.5980,
        "Random Forest":      0.6961,
        "XGBoost":            0.7961,
    },
    "Bioresponse": {
        "HRF Ultimate (GPU)": 0.8300,
        "SVM (RBF)":          0.7650,
        "Random Forest":      0.8200,
        "XGBoost":            0.8000,
    },
    "Higgs Boson": {
        "HRF Ultimate (GPU)": 0.7016,
        "SVM (RBF)":          0.6650,
        "Random Forest":      0.6866,
        "XGBoost":            0.6666,
    },
    "Magic Telescope": {
        "HRF Ultimate (GPU)": 0.8850,
        "SVM (RBF)":          0.8633,
        "Random Forest":      0.8833,
        "XGBoost":            0.8766,
    },
    "Musk v2": {
        "HRF Ultimate (GPU)": 1.0000,
        "SVM (RBF)":          0.9966,
        "Random Forest":      0.9983,
        "XGBoost":            1.0000,
    },
    "Satimage": {
        "HRF Ultimate (GPU)": 0.9416,
        "SVM (RBF)":          0.8816,
        "Random Forest":      0.9366,
        "XGBoost":            0.9233,
    },
    "Letter Recognition": {
        "HRF Ultimate (GPU)": 0.9250,
        "SVM (RBF)":          0.8633,
        "Random Forest":      0.9133,
        "XGBoost":            0.9033,
    },
}

results = []

hrf_wins = 0
svm_wins = 0
ties = 0

for dataset, scores in benchmark_results.items():

    hrf = scores["HRF Ultimate (GPU)"]
    svm = scores["SVM (RBF)"]

    difference = round((hrf - svm) * 100, 2)

    if hrf > svm:
        winner = "HRF"
        hrf_wins += 1

    elif svm > hrf:
        winner = "SVM"
        svm_wins += 1

    else:
        winner = "Tie"
        ties += 1

    results.append({
        "Dataset": dataset,
        "HRF Accuracy (%)": round(hrf * 100, 2),
        "SVM Accuracy (%)": round(svm * 100, 2),
        "Difference (%)": difference,
        "Winner": winner
    })

df = pd.DataFrame(results)

print("\n===== HRF vs SVM Comparison =====\n")
print(df)

script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.dirname(os.path.abspath(__file__))
df.to_csv(
    os.path.join(script_dir, "hrf_vs_svm_results.csv"),
    index=False
)

avg_hrf = (
    df["HRF Accuracy (%)"]
    .mean()
)

avg_svm = (
    df["SVM Accuracy (%)"]
    .mean()
)

print("\n===== SUMMARY =====\n")

print(f"HRF Wins : {hrf_wins}")
print(f"SVM Wins : {svm_wins}")
print(f"Ties     : {ties}")

print(f"\nAverage HRF Accuracy : {avg_hrf:.2f}%")
print(f"Average SVM Accuracy : {avg_svm:.2f}%")

print(
    f"\nAverage Improvement : "
    f"{avg_hrf - avg_svm:.2f}%"
)

best_dataset = df.loc[
    df["Difference (%)"].idxmax()
]

print("\nLargest HRF Advantage:")

print(
    f"{best_dataset['Dataset']} "
    f"({best_dataset['Difference (%)']}%)"
)